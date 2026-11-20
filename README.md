# Генератор РПД

Веб-приложение и набор CLI-инструментов для подготовки черновиков рабочих программ дисциплин (РПД). Проект собирает сведения из настроенных источников, координирует несколько профильных агентов через LangGraph и выдаёт документ, проверенный по JSON-схеме.

> Результат генерации — черновик для методической проверки. До утверждения необходимо сверить часы, ЗЕТ, компетенции, формы аттестации, учебный план и источники с официальными материалами вуза.

## Возможности

- подготовка РПД для одной дисциплины или комплекта дисциплин курса;
- подбор дисциплин по материалам учебного плана с возможностью вручную отредактировать список;
- работа трёх профильных узлов: University, Science и Business;
- поиск по индексам Yandex AI Studio и генерация структурированного ответа через Responses API;
- подбор свежих научных публикаций из Crossref и arXiv для учебных тем и практических заданий;
- правки к готовому черновику с вложениями TXT, DOCX и PDF;
- экспорт отдельной РПД в JSON и TXT, а комплекта — в ZIP-архив;
- отдельный пакет `rabota_it` для импорта разрешённых выгрузок ИТ-вакансий Работа.ру и локальной аналитики рынка.

## Быстрый старт

Нужен Python 3.11 или новее и доступ к Yandex AI Studio.

```powershell
git clone https://github.com/Kityra/rpd-generator.git
Set-Location rpd-generator
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Заполните `.env` значениями своего каталога Yandex Cloud, затем запустите интерфейс:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Откройте [http://localhost:8501](http://localhost:8501). Данные черновиков в этом интерфейсе хранятся только в текущей вкладке браузера, поэтому перед закрытием вкладки скачайте нужные файлы.

## Настройка

Скопируйте `.env.example` в `.env`. Файл `.env` исключён из Git: не добавляйте в репозиторий ключи, пароли или реальные идентификаторы окружения.

| Переменная | Назначение |
| --- | --- |
| `YANDEX_API_KEY` | ключ сервисного аккаунта Yandex Cloud |
| `YANDEX_FOLDER_ID` | идентификатор каталога Yandex Cloud |
| `YANDEX_MODEL_URI` | URI модели, формирующей итоговый черновик |
| `YANDEX_VECTOR_STORE_ID_RPD` | индекс с примерами РПД и учебными сведениями |
| `YANDEX_VECTOR_STORE_ID_BOOKS` | необязательный индекс учебных и научных материалов |
| `YANDEX_VECTOR_STORE_ID_HH` | необязательный индекс сведений о востребованных навыках |

Для профильных узлов можно отдельно задать `YANDEX_MODEL_URI_UNIVERSITY`, `YANDEX_MODEL_URI_SCIENCE` и `YANDEX_MODEL_URI_BUSINESS`. Если оставить их пустыми, используется Alice AI LLM Flash текущего каталога.

Перед первым запуском полезно проверить доступ к индексам:

```powershell
.\.venv\Scripts\python.exe check_yandex_indexes.py
.\.venv\Scripts\python.exe check_yandex_indexes.py --index rpd --query "Форма аттестации по дисциплине Базы данных"
.\.venv\Scripts\python.exe check_yandex_indexes.py --check-model
```

Подробные требования к ролям, индексам и ключам описаны в [YANDEX_AI_SETUP.md](YANDEX_AI_SETUP.md).

## Как устроена подготовка РПД

```text
Заявка преподавателя
        |
        v
  LangGraph-координатор
   /        |         \
University  Science  Business
   |           |         |
RPD/Books  Books +    HH
           Crossref,
           arXiv
        \    |    /
         Черновик РПД
              |
        Проверка схемой
              |
  правки → утверждение → экспорт
```

- **University** извлекает сведения о структуре РПД и учебные материалы.
- **Science** предлагает актуальные темы и задания по метаданным публикаций; полные тексты остаются у издателей и в arXiv.
- **Business** добавляет контекст востребованных навыков при наличии соответствующего индекса.
- Координатор формирует JSON по [`final_schema.json`](final_schema.json). Если обязательные официальные данные отсутствуют, интерфейс помечает их для сверки, а не подставляет как подтверждённые факты.

Подробнее о веб-интерфейсе — в [WEB_APP.md](WEB_APP.md), об устройстве и ограничениях пайплайна — в [PIPELINE_AUDIT.md](PIPELINE_AUDIT.md).

## Командная строка

Подготовить один черновик РПД в интерактивном режиме:

```powershell
.\.venv\Scripts\python.exe run_rpd.py
```

Либо передать сведения явно:

```powershell
.\.venv\Scripts\python.exe run_rpd.py `
  --discipline "Базы данных" `
  --institution "Южный федеральный университет" `
  --year "2026/2027" `
  --facts-file .\facts.txt `
  --output .\rpd_draft.json
```

Подготовить РПД по перечню дисциплин и сформировать читаемый отчёт:

```powershell
.\.venv\Scripts\python.exe run_course.py `
  --course "Прикладная информатика" `
  --institution "Южный федеральный университет" `
  --year "2026/2027" `
  --disciplines-file .\disciplines.txt `
  --output-dir .\course_output
```

В `disciplines.txt` укажите по одной дисциплине на строку. Если файл не передать, сервис попробует подобрать состав курса по индексу РПД; в обоих случаях требуется заполненный `.env`.

Провести независимый обзор свежих ИТ-публикаций:

```powershell
.\.venv\Scripts\python.exe run_science.py
.\.venv\Scripts\python.exe run_science.py --area security
```

### Вакансии Работа.ру

Пакет `rabota_it` работает только с полным JSON-снимком из разрешённой выгрузки или договорного HTTPS-источника. Он не собирает вакансии со страниц сайта.

```powershell
.\.venv\Scripts\python.exe -m rabota_it import .\vacancies.json
.\.venv\Scripts\python.exe -m rabota_it search "разработчик Python" --region Москва --technology Docker
.\.venv\Scripts\python.exe -m rabota_it summary "разработчик"
```

Формат входных данных, ограничения источника и использование пакета из LLM-компонента описаны в [rabota_it/README.md](rabota_it/README.md).

## Запуск в Docker

Создайте и заполните `.env`, затем выполните:

```powershell
docker compose up --build
```

Compose публикует контейнер только на `127.0.0.1:8501`. В текущем коде режим с сетевым адресом намеренно отключает форму генерации до подключения облачного входа. Для локальной демонстрации используйте команду Streamlit из раздела «Быстрый старт». Для внешнего доступа понадобятся отдельные настройка HTTPS и аутентификация; не открывайте демонстрационный интерфейс в интернет без них.

## Проверка проекта

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Структура репозитория

```text
app.py                 Streamlit-интерфейс
web_preview.py         сессионная очередь, правки и экспорт
rpd_graph.py           граф подготовки РПД и проверка схемы
run_rpd.py             CLI для одной дисциплины
run_course.py          CLI для комплекта дисциплин
science_literature.py  поиск и нормализация научных метаданных
rabota_it/             импорт и поиск по разрешённым выгрузкам вакансий
storage/, auth/,       компоненты устойчивого курса и адаптеры доступа
orchestration/
tests/                 автоматические тесты
```

## Документация

- [Подключение Yandex AI Studio](YANDEX_AI_SETUP.md)
- [Веб-приложение](WEB_APP.md)
- [Аудит пайплайна](PIPELINE_AUDIT.md)
- [Работа с выгрузками вакансий](rabota_it/README.md)

## Лицензия

Лицензия в репозитории пока не указана. До использования, распространения или развёртывания проекта в сторонней среде согласуйте условия с владельцем репозитория.
