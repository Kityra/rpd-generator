"""Look up article metadata for Science without adding files to Vector Store."""

from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from html import unescape
import re
from urllib.parse import quote
from xml.etree import ElementTree

import httpx

from orchestration.resilience import EXTERNAL, ExternalUnavailable, snapshot_notice
from rpd_specialists import SourceFragment


CROSSREF = "https://api.crossref.org/works"
ARXIV = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR = "https://api.semanticscholar.org/graph/v1/paper/batch"
ATOM = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
FRONTIER_WINDOW_DAYS = 365
SCIENCE_ERRORS = (ExternalUnavailable, httpx.HTTPError, ElementTree.ParseError, KeyError, TypeError, ValueError)
IT_AREAS = {
    "ai": (("0162-8828", "1939-3539", "1533-7928"), ("cs.AI", "cs.LG", "cs.CL")),
    "data": (("2150-8097", "2836-6573"), ("cs.DB",)),
    "security": (("1545-5971", "1941-0018", "2471-2566"), ("cs.CR",)),
    "systems": (("1045-9219", "0734-2071"), ("cs.DC", "cs.OS", "cs.NI")),
    "software": (("0098-5589", "1049-331X", "1557-7392"), ("cs.SE", "cs.PL")),
}


def normalise_doi(value: str) -> str:
    doi = re.sub(r"^(https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value.strip(), flags=re.IGNORECASE)
    return doi.lower() if re.fullmatch(r"10\.\d{4,9}/[^\s]+", doi) else ""


@dataclass(frozen=True)
class Article:
    title: str
    year: int | None
    url: str
    status: str
    doi: str = ""
    venue: str = ""
    authors: tuple[str, ...] = ()
    citations: int | None = None
    citation_source: str = ""
    area: str = ""
    abstract: str = ""
    cached_at: datetime | None = None

    def as_science_source(self, *, include_abstract: bool = False, include_snapshot_notice: bool = True) -> SourceFragment:
        text = (f"Название: {self.title}\nАвторы: {', '.join(self.authors)}\n"
                f"Год: {self.year or 'не указан'}\nИздание: {self.venue or 'не подтверждено'}\n"
                f"Статус: {self.status}\nОбласть: {self.area}\nСсылка: {self.url}")
        if include_abstract and self.abstract:
            text += f"\nФрагмент аннотации: {self.abstract}"
        if include_snapshot_notice and self.cached_at is not None:
            text += "\n" + snapshot_notice(self.citation_source or "arXiv", self.cached_at)
        return {
            "file": self.url,
            "text": text,
        }


def _get(client: httpx.Client, service: tuple[str, str], url: str, params: dict) -> httpx.Response:
    def request(timeout):
        response = client.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response

    return EXTERNAL.call(service, request, budget=12, read=7)


def _article_snapshot(service, key, fetch, notices):
    articles, taken_at = EXTERNAL.search((service, *key), lambda: tuple(fetch()), max_age=6 * 3600)
    if taken_at is not None:
        notices.append(snapshot_notice(service, taken_at))
        return [replace(article, cached_at=taken_at) for article in articles]
    return list(articles)


def crossref_articles(
    client: httpx.Client, area: str | None, since: date, today: date, *, query: str = "",
) -> list[Article]:
    issns = IT_AREAS[area][0] if area else ()
    filters = [*(f"issn:{issn}" for issn in issns), "type:journal-article", "type:proceedings-article",
               f"from-pub-date:{since}", f"until-pub-date:{today}"]
    params = {
        "filter": ",".join(filters),
        "sort": "is-referenced-by-count", "order": "desc",
        "select": "DOI,title,container-title,issued,author,is-referenced-by-count,update-to",
        "rows": 6,
    }
    if query:
        params.update({"query.bibliographic": query, "sort": "score", "rows": 12})
        params["select"] += ",abstract"
    response = _get(client, ("Crossref", "search"), CROSSREF, params)
    papers = []
    for work in response.json()["message"]["items"]:
        title = next(iter(work.get("title") or []), "").strip()
        doi = normalise_doi(work.get("DOI") or "")
        updates = work.get("update-to") or []
        if not title or not doi or any("retract" in str(update.get("type", "")).lower() for update in updates):
            continue
        date_parts = (work.get("issued") or {}).get("date-parts") or [[]]
        year = date_parts[0]
        papers.append(Article(
            title=title[:500],
            year=year[0] if year else None,
            url="https://doi.org/" + quote(doi, safe="/"),
            status="published_candidate",
            doi=doi,
            venue=next(iter(work.get("container-title") or []), "")[:200],
            authors=tuple(
                name for person in (work.get("author") or [])[:8]
                if (name := " ".join(person.get(key) or "" for key in ("given", "family")).strip())
            ),
            citations=work.get("is-referenced-by-count"),
            citation_source="Crossref",
            area=area or "",
            abstract=" ".join(unescape(re.sub(r"<[^>]+>", " ", work.get("abstract") or "")).split())[:800],
        ))
    return papers


def has_retraction(client: httpx.Client, doi: str) -> bool:
    response = _get(client, ("Crossref", "retraction"), CROSSREF,
                    {"filter": f"updates:{doi},update-type:retraction", "rows": 0})
    return response.json()["message"]["total-results"] > 0


def _retraction_status(client: httpx.Client, doi: str) -> tuple[bool, bool]:
    result, taken_at = EXTERNAL.search(("Crossref", "retraction", doi),
                                      lambda: (has_retraction(client, doi),), max_age=6 * 3600)
    return result[0], taken_at is None


def arxiv_articles(
    client: httpx.Client, areas: tuple[str, ...], since: date, today: date, *, terms: tuple[str, ...] = (),
) -> list[Article]:
    categories = {category: area for area in areas for category in IT_AREAS[area][1]}
    scope = " OR ".join(f"cat:{category}" for category in categories)
    if terms:
        keywords = " OR ".join(f"ti:{term}" for term in terms)
        scope = f"({scope}) AND ({keywords}) AND submittedDate:[{since:%Y%m%d}0000 TO {today:%Y%m%d}2359]"
    response = _get(client, ("arXiv", "search"), ARXIV, {
        "search_query": scope,
        "sortBy": "relevance" if terms else "submittedDate",
        "sortOrder": "descending",
        "max_results": 8,
    })
    feed = ElementTree.fromstring(response.content)
    if feed.tag != f"{{{ATOM['atom']}}}feed":
        raise ValueError("Неверный формат ленты arXiv.")
    papers = []
    for entry in feed.findall("atom:entry", ATOM):
        title = " ".join(entry.findtext("atom:title", "", ATOM).split())
        raw_url = entry.findtext("atom:id", "", ATOM).strip()
        if "arxiv.org/api/errors" in raw_url:
            raise ValueError("arXiv вернул ошибку в ленте.")
        if not title or not re.fullmatch(r"https?://arxiv\.org/abs/[\w./-]+", raw_url):
            continue
        try:
            published = date.fromisoformat(entry.findtext("atom:published", "", ATOM)[:10])
        except ValueError:
            continue
        if not since <= published <= today:
            continue
        codes = [category.get("term", "") for category in entry.findall("atom:category", ATOM)]
        area = next((categories[code] for code in codes if code in categories), "")
        if not area:
            continue
        papers.append(Article(
            title=title[:500], year=published.year,
            url=raw_url.replace("http://", "https://", 1), status="arxiv_candidate",
            doi=normalise_doi(entry.findtext("arxiv:doi", "", ATOM)),
            authors=tuple(author.findtext("atom:name", "", ATOM) for author in entry.findall("atom:author", ATOM)[:8]),
            area=area,
            abstract=" ".join(entry.findtext("atom:summary", "", ATOM).split())[:800] if terms else "",
        ))
    return papers


def find_course_articles(query: str, *, verified_dois: set[str]) -> tuple[list[Article], list[str]]:
    """Find recent works for a particular discipline, with bounded optional lookups."""
    ignored = {"and", "for", "with", "from", "the", "using"}
    terms = tuple(dict.fromkeys(word for word in re.findall(r"[a-z][a-z0-9]+", query.lower())
                               if word not in ignored))[:8]
    if len(terms) < 2:
        return [], []

    def match(paper: Article) -> int:
        words = {word.removesuffix("s") for word in re.findall(r"[a-z][a-z0-9]+", paper.title.lower())}
        return sum(term.removesuffix("s") in words for term in terms)

    today = date.today()
    since = today - timedelta(days=FRONTIER_WINDOW_DAYS)
    unavailable = []
    selected = []
    with httpx.Client(timeout=7, headers={"User-Agent": "RPD-Science/0.1"}) as client:
        try:
            published = _article_snapshot("Crossref", (None, since, today, terms),
                lambda: crossref_articles(client, None, since, today, query=" ".join(terms)), unavailable)
        except SCIENCE_ERRORS:
            published = []
            unavailable.append("Crossref")
        ranked = sorted((paper for paper in published if match(paper) >= 2),
                        key=lambda paper: (paper.doi in verified_dois, match(paper), paper.citations or 0),
                        reverse=True)
        unique = {}
        for paper in ranked:
            unique.setdefault(paper.doi, paper)
        for paper in list(unique.values())[:3]:
            try:
                retracted, current = _retraction_status(client, paper.doi)
                if retracted:
                    continue
            except SCIENCE_ERRORS:
                unavailable.append("Crossref: проверка отзыва")
                break
            if not current and "Crossref: проверка отзыва по снимку" not in unavailable:
                unavailable.append("Crossref: проверка отзыва по снимку")
            selected.append(replace(paper, status="verified" if current and paper.doi in verified_dois
                                    else "published_candidate"))
        try:
            areas = tuple(IT_AREAS)
            preprints = _article_snapshot("arXiv", (areas, since, today, terms),
                lambda: arxiv_articles(client, areas, since, today, terms=terms), unavailable)
        except SCIENCE_ERRORS:
            preprints = []
            unavailable.append("arXiv")

    known_titles = {paper.title.casefold() for paper in selected}
    for paper in sorted(preprints, key=match, reverse=True):
        if match(paper) >= 2 and paper.doi not in unique and paper.title.casefold() not in known_titles:
            selected.append(paper)
            break
    return selected, unavailable


def semantic_citations(client: httpx.Client, dois: list[str], key: str) -> dict[str, int]:
    response = client.post(
        SEMANTIC_SCHOLAR, headers={"x-api-key": key}, params={"fields": "externalIds,citationCount"},
        json={"ids": [f"DOI:{doi}" for doi in dois]},
    )
    response.raise_for_status()
    counts = {}
    for paper in response.json():
        if not isinstance(paper, dict):
            continue
        doi = normalise_doi((paper.get("externalIds") or {}).get("DOI") or "")
        count = paper.get("citationCount")
        if doi and isinstance(count, int) and count >= 0:
            counts[doi] = count
    return counts


def select_articles(published: list[Article], preprints: list[Article]) -> list[Article]:
    """Prefer verified papers and diversity across IT areas."""
    selected: list[Article] = []
    dois: set[str] = set()
    titles: set[str] = set()

    def add(candidates: list[Article], limit: int) -> None:
        added = 0
        for article in candidates:
            if added == limit or len(selected) >= 5:
                break
            title = " ".join(re.findall(r"\w+", article.title.casefold()))
            if (article.doi and article.doi in dois) or title in titles:
                continue
            selected.append(article)
            titles.add(title)
            if article.doi:
                dois.add(article.doi)
            added += 1

    add([paper for paper in published if paper.status == "verified"], 4)
    add(preprints, 1)
    for area in dict.fromkeys(paper.area for paper in published):
        if any(card.area == area for card in selected):
            continue
        ranked = sorted((paper for paper in published if paper.area == area), key=lambda paper: -(paper.citations or 0))
        add(ranked, 1)
    add(sorted(published, key=lambda paper: -(paper.citations or 0)), 5)
    add(preprints, 5)
    return selected


def search_frontier(
    area: str | None = None, *, verified_dois: set[str], semantic_key: str = "",
) -> tuple[list[Article], list[str]]:
    """Return up to five candidate cards and names of unavailable sources."""
    unavailable = []
    areas = (area,) if area else tuple(IT_AREAS)
    today = date.today()
    since = today - timedelta(days=FRONTIER_WINDOW_DAYS)
    with httpx.Client(timeout=12, headers={"User-Agent": "RPD-Science/0.1"}) as client:
        published = []
        for current_area in areas:
            try:
                published.extend(_article_snapshot("Crossref", (current_area, since, today, ()),
                    lambda: crossref_articles(client, current_area, since, today), unavailable))
            except SCIENCE_ERRORS:
                unavailable.append(f"Crossref ({current_area})")

        try:
            preprints = _article_snapshot("arXiv", (areas, since, today, ()),
                lambda: arxiv_articles(client, areas, since, today), unavailable)
        except SCIENCE_ERRORS:
            unavailable.append("arXiv")
            preprints = []

        checked = []
        seen_dois: set[str] = set()
        retracted_dois: set[str] = set()
        remaining_checks = 5
        for paper in published:
            if paper.doi in seen_dois:
                continue
            seen_dois.add(paper.doi)
            if paper.doi in verified_dois and remaining_checks:
                remaining_checks -= 1
                try:
                    retracted, current = _retraction_status(client, paper.doi)
                    if retracted:
                        retracted_dois.add(paper.doi)
                        continue
                    if current:
                        paper = replace(paper, status="verified")
                    elif "Crossref: проверка отзыва по снимку" not in unavailable:
                        unavailable.append("Crossref: проверка отзыва по снимку")
                except SCIENCE_ERRORS:
                    if "Crossref: проверка отзыва" not in unavailable:
                        unavailable.append("Crossref: проверка отзыва")
            checked.append(paper)

        cards = select_articles(checked, [paper for paper in preprints if paper.doi not in retracted_dois])
        counts = {}
        dois = [card.doi for card in cards if card.status != "arxiv_candidate"]
        if semantic_key and dois:
            try:
                counts = semantic_citations(client, dois, semantic_key)
            except (httpx.HTTPError, KeyError, TypeError, ValueError):
                unavailable.append("Semantic Scholar")

    enriched = [
        replace(card, citations=counts[card.doi], citation_source="Semantic Scholar")
        if card.status != "arxiv_candidate" and card.doi in counts else card
        for card in cards
    ]
    return enriched, unavailable
