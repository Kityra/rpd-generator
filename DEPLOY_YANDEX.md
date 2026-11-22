# Развёртывание rpd-generator на виртуальной машине Yandex Cloud

Инструкция рассчитана на Linux-ВМ с Docker Engine и плагином Docker Compose. Веб-приложение, фоновый обработчик и планировщик используют один образ. Сервис `migrate` применяет миграции до их запуска. PostgreSQL хранит курсы и контрольные точки. Веб-приложение открывается по HTTP на порту 8080.

## 1. Подготовить ВМ

1. Подключитесь по SSH и установите Docker Engine с плагином Compose. Проверьте команды ниже. Если пользователю требуется sudo, выполняйте Docker-команды с sudo. [Установка для Ubuntu](https://docs.docker.com/engine/install/ubuntu/) · [Подключение к ВМ](https://yandex.cloud/ru/docs/compute/operations/vm-connect/ssh)
2. Если используете домен, направьте его DNS-запись A на публичный IPv4-адрес ВМ. Можно открывать приложение и по IP.
3. В группе безопасности разрешите входящий TCP 22 с адресов администраторов и TCP 8080 с адресов тестировщиков. Разрешите исходящий HTTPS для загрузки образов и обращения к Yandex AI Studio и поставщикам данных. PostgreSQL доступен на ВМ через `127.0.0.1:5432`. [Группы безопасности Yandex Cloud](https://yandex.cloud/ru/docs/vpc/concepts/security-groups)

~~~bash
docker version
docker compose version
~~~

## 2. Перенести код и заполнить .env

Передайте код по SSH или клонируйте репозиторий. Выполняйте команды из папки проекта с `compose.yaml` и `Dockerfile`. Сохраняйте эту папку при обновлениях, чтобы Compose использовал прежние тома.

~~~bash
umask 077
test -f .env || cp .env.example .env
chmod 600 .env
~~~

| Переменные в .env | Что указать |
| --- | --- |
| RPD_PUBLIC_ORIGIN | Адрес, который открывают тестировщики, например `http://rpd.example.org:8080`. Подставьте IP ВМ, если используете его вместо домена. Адрес используется в ссылках-приглашениях. |
| RPD_WEB_PORT | Внешний порт приложения. Значение по умолчанию — `8080`. Для `8000` также укажите этот порт в `RPD_PUBLIC_ORIGIN` и в группе безопасности. |
| POSTGRES_PASSWORD | Новый случайный пароль из шестнадцатеричных символов. Compose использует его и при создании БД, и в строке подключения. |
| SESSION_SIGNING_KEY | Для первой установки создайте отдельный случайный ключ длиной не менее 32 символов. Сохраняйте его при обновлениях на том же домене. |
| YANDEX_API_KEY, YANDEX_FOLDER_ID, YANDEX_MODEL_URI | Настройки действующего сервисного аккаунта и модели. |
| TEACHERS_INDEX_ID, EXPERTS_INDEX_ID | Индексы методистов и экспертов в Yandex AI Studio. Управление программами доступно в кабинете методиста. Если используется прежний дополнительный справочник PROGRAM_HEADS_INDEX_ID, сохраните его для существующих учётных записей. |
| YANDEX_VECTOR_STORE_ID_RPD и остальные индексы | Перенесите рабочие значения RPD, Books, HH, Priklad и Interview. Поддерживаются прежние альтернативные имена из .env.example. |

Пароль БД и ключ сессий можно сгенерировать двумя вызовами `openssl rand -hex 32`. Заполните `.env` на ВМ или передайте его по SSH. Compose направляет `APP_DATABASE_URL` на внутренний сервис `postgres`, а пустой `CHECKPOINT_URL` направляет LangGraph в ту же базу. В контейнерах включён режим `production` с входом через настроенные справочники пользователей. Для доступа к сохранённым курсам используйте прежние учётные записи владельцев.

~~~bash
docker compose config --quiet
~~~

## 3. Восстановить PostgreSQL с курсами

Выполните этот раздел, если нужен перенос текущих курсов. Для первой установки с пустой базой переходите к разделу 4.

Используйте свежий архив всей базы rpd в формате pg_dump --format=custom. Он включает курсы, версии, прикладные данные, задания воркера и контрольные точки LangGraph из общей БД. Файлы Yandex AI Studio продолжают храниться там же и подключаются через идентификаторы индексов. Храните дамп отдельно от исходников и передавайте по SSH. [Резервное копирование PostgreSQL](https://www.postgresql.org/docs/16/app-pgdump.html)

Сначала запустите только PostgreSQL. Следующая команда проверки должна показать false. Если она показывает true, целевая база уже инициализирована. Для переноса в новый экземпляр восстановите архив в отдельную пустую базу перед первым запуском веб-приложения. pg_restore выполняется одной транзакцией и прерывается при ошибке. [Документация pg_restore](https://www.postgresql.org/docs/16/app-pgrestore.html)

~~~bash
docker compose up -d --wait postgres
docker compose exec -T postgres psql -U rpd -d rpd -c \
  "SELECT to_regclass('public.courses') IS NOT NULL AS database_already_initialized;"
docker compose exec -T postgres pg_restore -U rpd -d rpd \
  --exit-on-error --single-transaction --no-owner --no-privileges < /path/to/rpd.dump
~~~

## 4. Запустить приложение и проверить

~~~bash
docker compose up -d
docker compose ps -a
docker compose logs --tail=60 migrate app worker scheduler
curl -fsS http://localhost:8080/_stcore/health
~~~

С компьютера тестировщика откройте адрес из `RPD_PUBLIC_ORIGIN`. Проверьте вход в `/teacher` и `/expert`, список перенесённых курсов, «Управление программами» и генерацию черновика. При обновлении кода выполните `docker compose up -d --build` из той же папки. Для остановки используйте `docker compose stop`, чтобы сохранить данные в томах.
