"""Look up article metadata for Science without adding files to Vector Store."""

from dataclasses import dataclass, replace
from datetime import date, timedelta
import re
from time import sleep
from urllib.parse import quote
from xml.etree import ElementTree

import httpx

from rpd_specialists import SourceFragment


CROSSREF = "https://api.crossref.org/works"
ARXIV = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR = "https://api.semanticscholar.org/graph/v1/paper/batch"
ATOM = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
FRONTIER_WINDOW_DAYS = 365
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

    def as_science_source(self) -> SourceFragment:
        text = (f"Название: {self.title}\nАвторы: {', '.join(self.authors)}\n"
                f"Год: {self.year or 'не указан'}\nИздание: {self.venue or 'не подтверждено'}\n"
                f"Статус: {self.status}\nОбласть: {self.area}\nСсылка: {self.url}")
        return {
            "file": self.url,
            "text": text,
        }


def crossref_articles(client: httpx.Client, area: str, since: date, today: date) -> list[Article]:
    issns, _ = IT_AREAS[area]
    filters = [*(f"issn:{issn}" for issn in issns), "type:journal-article", "type:proceedings-article",
               f"from-pub-date:{since}", f"until-pub-date:{today}"]
    response = client.get(CROSSREF, params={
        "filter": ",".join(filters),
        "sort": "is-referenced-by-count", "order": "desc",
        "select": "DOI,title,container-title,issued,author,is-referenced-by-count,update-to",
        "rows": 6,
    })
    response.raise_for_status()
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
            area=area,
        ))
    return papers


def has_retraction(client: httpx.Client, doi: str) -> bool:
    response = client.get(CROSSREF, params={"filter": f"updates:{doi},update-type:retraction", "rows": 0})
    response.raise_for_status()
    return response.json()["message"]["total-results"] > 0


def arxiv_articles(client: httpx.Client, areas: tuple[str, ...], since: date, today: date) -> list[Article]:
    categories = {category: area for area in areas for category in IT_AREAS[area][1]}
    scope = " OR ".join(f"cat:{category}" for category in categories)
    response = client.get(ARXIV, params={
        "search_query": scope,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": 8,
    }, timeout=12)
    response.raise_for_status()
    papers = []
    for entry in ElementTree.fromstring(response.content).findall("atom:entry", ATOM):
        title = " ".join(entry.findtext("atom:title", "", ATOM).split())
        raw_url = entry.findtext("atom:id", "", ATOM).strip()
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
        ))
    return papers


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
                published.extend(crossref_articles(client, current_area, since, today))
            except (httpx.HTTPError, KeyError, TypeError, ValueError):
                unavailable.append(f"Crossref ({current_area})")

        try:
            preprints = arxiv_articles(client, areas, since, today)
        except (httpx.HTTPError, ElementTree.ParseError) as error:
            unavailable.append("arXiv")
            preprints = []
            may_retry = not isinstance(error, httpx.HTTPStatusError) or error.response.status_code >= 500
            if len(areas) > 1 and may_retry:
                sleep(3)
                try:
                    preprints = arxiv_articles(client, areas[:1], since, today)
                    unavailable[-1] = "arXiv (часть категорий)"
                except (httpx.HTTPError, ElementTree.ParseError):
                    pass

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
                    if has_retraction(client, paper.doi):
                        retracted_dois.add(paper.doi)
                        continue
                    paper = replace(paper, status="verified")
                except (httpx.HTTPError, KeyError, TypeError, ValueError):
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
