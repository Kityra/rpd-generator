# Развёртывание rpd-generator на виртуальной машине Yandex Cloud

Инструкция рассчитана на Linux-ВМ с Docker Engine и плагином Docker Compose. Веб-приложение, фоновый обработчик и планировщик используют один образ. Сервис migrate применяет миграции до запуска приложения, PostgreSQL хранит курсы и контрольные точки, Caddy завершает HTTPS.

## 1. Подготовить ВМ и домен

1. Подключитесь по SSH и установите Docker Engine с плагином Compose. Проверьте команды ниже. Если пользователю требуется sudo, выполняйте Docker-команды с sudo. [Установка для Ubuntu](https://docs.docker.com/engine/install/ubuntu/) · [Подключение к ВМ](https://yandex.cloud/ru/docs/compute/operations/vm-connect/ssh)
2. Направьте DNS-запись A домена на публичный IPv4-адрес ВМ. Запись AAAA нужна только при работающем доступе по IPv6.
3. В группе безопасности разрешите входящий TCP 22 с адресов администраторов и TCP 80 и 443 для HTTPS и выпуска сертификата. Разрешите исходящий HTTPS для образов, Yandex AI Studio и поставщиков данных. Порты PostgreSQL и Streamlit в Compose привязаны к 127.0.0.1 ВМ. [Группы безопасности Yandex Cloud](https://yandex.cloud/ru/docs/vpc/concepts/security-groups) · [Автоматический HTTPS Caddy](https://caddyserver.com/docs/quick-starts/https)

~~~bash
docker version
docker compose version
~~~

## 2. Перенести код и заполнить .env

Передайте архив кода по SSH или возьмите ревизию репозитория, в которой есть оба Compose-файла, Dockerfile и код последней версии проекта. Распакуйте в отдельную папку и выполняйте следующие команды из её корня. Папку проекта сохраняйте при последующих обновлениях, чтобы имя проекта Compose и связанные с ним тома оставались прежними.

~~~bash
umask 077
test -f .env || cp .env.example .env
chmod 600 .env
~~~

| Переменные в .env | Что указать |
| --- | --- |
| RPD_DOMAIN | Домен без схемы и пути, например rpd.example.org. Конфигурация HTTPS задаст PUBLIC_ORIGIN и включит production. |
| POSTGRES_PASSWORD | Новый случайный пароль из шестнадцатеричных символов. Compose использует его и при создании БД, и в строке подключения. |
| SESSION_SIGNING_KEY | Для первой установки создайте отдельный случайный ключ длиной не менее 32 символов. Сохраняйте его при обновлениях на том же домене. |
| YANDEX_API_KEY, YANDEX_FOLDER_ID, YANDEX_MODEL_URI | Настройки действующего сервисного аккаунта и модели. |
| TEACHERS_INDEX_ID, EXPERTS_INDEX_ID, PROGRAM_HEADS_INDEX_ID | Индексы пользователей Yandex AI Studio. Для доступа руководителей к кабинету /head заполните отдельный индекс руководителей. Для методиста и эксперта нужны оба их индекса. |
| YANDEX_VECTOR_STORE_ID_RPD и остальные индексы | Перенесите рабочие значения RPD, Books, HH, Priklad и Interview. Поддерживаются прежние альтернативные имена из .env.example. |

Пароль БД и ключ сессий можно сгенерировать двумя вызовами `openssl rand -hex 32`. Заполните .env на ВМ или передайте его напрямую по SSH. Контейнеры получают APP_DATABASE_URL с внутренним именем postgres, а пустой CHECKPOINT_URL направляет LangGraph в ту же базу. Локальный демонстрационный справочник отключается конфигурацией HTTPS. На новом домене нужно войти заново. Доступ к уже сохранённым курсам определяется идентификаторами владельцев, поэтому используйте прежние учётные записи из соответствующего справочника Yandex.

~~~bash
docker compose -f compose.yaml -f compose.https.yaml config --quiet
~~~

## 3. Восстановить PostgreSQL с курсами

Выполните этот раздел, если нужен перенос текущих курсов. Для первой установки с пустой базой переходите к разделу 4.

Используйте свежий архив всей базы rpd в формате pg_dump --format=custom. Он включает курсы, версии, прикладные данные, задания воркера и контрольные точки LangGraph из общей БД. Файлы Yandex AI Studio продолжают храниться там же и подключаются через идентификаторы индексов. Храните дамп отдельно от исходников и передавайте по SSH. [Резервное копирование PostgreSQL](https://www.postgresql.org/docs/16/app-pgdump.html)

Сначала запустите только PostgreSQL. Следующая команда проверки должна показать false. Если она показывает true, целевая база уже инициализирована. Для переноса в новый экземпляр восстановите архив в отдельную пустую базу перед первым запуском веб-приложения. pg_restore выполняется одной транзакцией и прерывается при ошибке. [Документация pg_restore](https://www.postgresql.org/docs/16/app-pgrestore.html)

~~~bash
docker compose -f compose.yaml -f compose.https.yaml up -d --wait postgres
docker compose exec -T postgres psql -U rpd -d rpd -c \
  "SELECT to_regclass('public.courses') IS NOT NULL AS database_already_initialized;"
docker compose exec -T postgres pg_restore -U rpd -d rpd \
  --exit-on-error --single-transaction --no-owner --no-privileges < /path/to/rpd.dump
~~~

## 4. Запустить приложение и проверить

~~~bash
docker compose -f compose.yaml -f compose.https.yaml up -d --build --wait
docker compose -f compose.yaml -f compose.https.yaml ps -a
docker compose -f compose.yaml -f compose.https.yaml logs --tail=60 migrate worker scheduler caddy
curl -fsS https://rpd.example.org/_stcore/health
~~~

Подставьте реальный домен. Проверьте вход в /teacher и /expert, список перенесённых курсов и генерацию нового черновика. При настроенном индексе руководителей проверьте /head. При обновлении кода повторите запуск с --build из той же папки. Для остановки используйте stop, чтобы тома и сертификаты остались на месте.
