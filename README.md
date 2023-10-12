# kgqa

> This repository contains the source code for `SemantIQ`, a system that allows formulating structured queries with loose natural language constraints, as well as for it's query language, `SemantIQ-QL`.

## Querying via SemantIQ

Once the project is [set up](#getting-started), simply run [`kgqa.py`](kgqa.py).
```sh
python3 kgqa.py
> X: director(X, "Quentin Tarantino")
```

## Getting Started

This project queries the [wikidata](https://www.wikidata.org/wiki/Wikidata:Main_Page) dataset. A brief overview of the dataset can be found [here](https://gist.github.com/anton5798/68c1975eb661abd76d68493bc98d6099). To set up the corresponding database tables proceed as follows:

1. Download and decompress the wikidata dump from [here](https://dumps.wikimedia.org/wikidatawiki/entities/).

```sh
curl https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.gz --output dump.json.gz
gzip -d jump.json.gz
```

2. Use the [`migrador.rb`](https://bitbucket.org/danielhz/wikidata-experiments/raw/9fb724eb90fdc242434db8fd36d88950eb2255c0/postgresql-experiment-scripts/load-data/migrador.rb) script provided by [wikidata-experiments](https://bitbucket.org/danielhz/wikidata-experiments/src/master/) to convert the dump into the required format.

```sh
curl https://bitbucket.org/danielhz/wikidata-experiments/raw/9fb724eb90fdc242434db8fd36d88950eb2255c0/postgresql-experiment-scripts/load-data/migrador.rb --output migrador.rb
mkdir csv
ruby migrador.rb
```

3. Set up a `wikidata` database in [`postgres`](https://www.postgresql.org) with `UTF-8` encoding.

```postgres
CREATE DATABASE wikidata WITH encoding = 'UTF8';
```

4. Populate the database using the [`sql/setup.sql`](sql/setup.sql) script.

```sh
psql -U $PSQL_USERNAME -d wikidata -f sql/setup.sql
```

5. Create and customize the configuration file `./config.yaml`. See [`config.template.yaml`](config.template.yaml) for the required parameters.

```sh
cp config.template.yaml config.yaml
```

In particular, this requires configuring the following parameters:

| **Config Parameter**          | **Description**                                      |
| ----------------------------- | ---------------------------------------------------- |
| `psql.username`               | `postgres` username                                  |
| `psql.password`               | `postgres` password                                  |
| `language_model.open_api_key` | [OpenAI API Key](https://openai.com/blog/openai-api) |

6. To install the project's dependencies execute the following command.
```sh
pip3 install -r requirements.txt
```

7. Next, generate the required embeddings via the provided setup script.
```sh
python3 setup.py ComputeEmbeddings
```

8. Finally, populate the `claims` table with *invertible* predicates by run supplied script.

```sh
python3 setup.py InvertPredicates
```

## Bugs

If you experience bugs, or have suggestions for improvements, please use the issue tracker to report them.
