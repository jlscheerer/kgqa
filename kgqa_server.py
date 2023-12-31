import threading
import time
import requests
import json
import re
import itertools

from flask import Flask, request
from flask_cors import CORS, cross_origin
import uuid

import networkx as nx

from kgqa.Database import Database
from kgqa.PromptBuilder import PromptBuilder
from kgqa.QueryParser import QueryParser, Variable
from kgqa.QueryGraph import QueryGraphAggregate, QueryGraphConstantNode, QueryGraphEntityConstantNode, QueryGraphFilter, QueryGraphGeneratedNode, QueryGraphPropertyConstantNode, QueryGraphPropertyEdge, QueryGraphPropertyNode, QueryGraphSerializer, QueryGraphVariableNode, query2aqg, aqg2wqg
from kgqa.SPARQLBackend import wqg2sparql
from kgqa.PostProcessing import run_and_rank
from kgqa.sparql2sql import sparql2sql

search_uuid_status = dict()
search_uuid_results = dict()

def retrieve_wiki_image(image):
    image = image.replace(" ", "_")
    image = f"File:{image}"
    response = requests.get(f"https://en.wikipedia.org/w/api.php?action=query&titles={image}&prop=imageinfo&iiprop=url&format=json")
    wiki = next(iter(json.loads(response.text)["query"]["pages"].values()))
    try:
        return wiki["imageinfo"][0]["url"]
    except:
        return None


def parse_result(result):
    if not isinstance(result, str):
        # NOTE relevant for aggregates.
        return result, result, result
    # TODO(jlscheerer) Hack. Clean this up!
    match = re.findall("^(.*?) \\((.*?)\\) \\[f=(.*?)\\]$", result)[0]
    label, id_, score = match[0], match[1], match[2]
    return label, id_, score

def format_key(key):
    print("key", key)
    # TODO(jlscheerer) Hack. Clean this up!
    match = re.findall("^(.*?) \\((PID|QID)\\)", key)
    return match[0][0]

def format_value(value):
    label, _, score = parse_result(value)
    return f"{label} ({float(score):.02f})"

def escape_ansi(line):
    ansi_escape = re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', line)

def handle_search_request(job):
    try:
        print("handle_search_request", job)
        query, id_, timestamp = job["query"], job["uuid"], job["timestamp"]

        try:
            pq = QueryParser().parse(query)
        except:
            search_uuid_status[id_] = "Generating Intermediate Representation"
            QirK = PromptBuilder(template="QUERY_GENERATE").set("QUERY", query).execute()

            search_uuid_status[id_] = "Parsing Query"
            pq = QueryParser().parse(QirK)

        search_uuid_status[id_] = "Generating Abstract Query Graph"
        aqg = query2aqg(pq)

        search_uuid_status[id_] = "Synthesizing Executable Query Graph"
        wqg = aqg2wqg(aqg)

        search_uuid_status[id_] ="Emitting SPARQL Code"
        qs = wqg2sparql(wqg)

        # TODO(jlscheerer) We now perform the conversion twice
        sql = sparql2sql(qs)

        search_uuid_status[id_] = "Executing Query on Wikidata"
        results, columns = run_and_rank(qs, wqg)

        # assert len(pq.head.items) == 1
        # head = pq.head.items[0]
        # assert isinstance(head, Variable)
        # result_type = head.type_info()

        """
        target_column = None
        for column in columns:
            if column == head.name:
                target_column = column
        """

        print(columns)
        print(pq.head.items)

        head_names = [head.name for head in pq.head.items]
        result_ids = []
        query_results = []

        print("head_names", head_names)

        for row in results.to_dict(orient="records"):
            score = float(row["Score"])
            derivation = [f"{format_key(key)} ↦ {format_value(value)}" for key, value in row.items() if key not in [*head_names, "Score"]]

            head_data = []
            for head in pq.head.items:
                result_type = head.type_info()
                label, rid, _ = parse_result(row[head.name])
                if result_type == "entity_id":
                    query_id_ = rid
                    title = label
                    link = f"entity?id={rid}"
                elif result_type == "date" or result_type == "string" or result_type == "numeric":
                    # TODO(jlscheerer) Show unit or calendar type.
                    title = rid
                    query_id_ = "None"
                    link = None
                else:
                    raise AssertionError
                if result_type == "entity_id" and query_id_ is not None:
                    result_ids.append(query_id_)

                head_data.append({
                    "variable": head.name,
                    "type": result_type,
                    "id": query_id_,
                    "title": title,
                    "link": link,
                })

            query_results.append(
                {
                    "head": head_data,
                    "scoreColor": ["#E3EDD5", "#FBE7CD"][score < 1],
                    "score": f"{score:.02f}",
                    "opacity": int((score ** 3) * 100),
                    "derivation": ", ".join(derivation)
                }
            )
        
        tok = time.time_ns()
        seconds = (tok - timestamp) / 10**9

        db = Database()
        db._conn.set_isolation_level(0)
        result_ids = ", ".join([f"'{rid}'" for rid in result_ids])

        if len(result_ids) > 0:
            images = {x[0]: x[1] for x in db.fetchall(f"""
            SELECT *
            FROM (
                    SELECT c.entity_id, c.datavalue_string, q.datavalue_string,
                        (ROW_NUMBER() OVER (PARTITION BY c.entity_id ORDER BY SUBSTRING(id FROM '[0-9]+$')::int)) AS n
                    FROM claims_inv c, qualifiers q
                    WHERE c.entity_id IN ({result_ids}) AND c.property = 'P18'
                    AND q.claim_id = c.id AND q.qualifier_property = 'P2096'
                ) AS images
            WHERE n <= 1
            """)}
        else:
            images = dict()

        if len(result_ids) > 0:
            descriptions = {x[0]: x[1] for x in db.fetchall(f"""
            SELECT id, value FROM descriptions_en WHERE id IN ({result_ids})
            """)}
        else:
            descriptions = dict()

        for result in query_results:
            for item in result["head"]:
                if item["id"] in descriptions:
                    item["description"] = descriptions[item["id"]]
                if item["id"] in images:
                    if len(result_ids) < 50:
                        item["image"] = retrieve_wiki_image(images[item["id"]])

        nodes = []
        for i, node in enumerate(wqg.nodes):
            data = {"id": str(node.id_.value), "data": {"label": str(node.id_.value)}, "type": "variableNode"}
            if isinstance(node, QueryGraphPropertyNode):
                # TODO(jlscheerer) Maybe we can visualize these also?
                data["data"]["label"] = f"{node.property} (property)"
                continue
            elif isinstance(node, QueryGraphVariableNode):
                data["data"]["label"] = f"{node.variable.name} (variable)"
            elif isinstance(node, QueryGraphEntityConstantNode):
                data["data"]["label"] = f"'{node.constant.value}' (entity)"
            elif isinstance(node, QueryGraphPropertyConstantNode):
                # TODO(jlscheerer) Maybe we can visualize these also?
                data["data"]["label"] = f"'{node.constant}' (property)"
                continue
            elif isinstance(node, QueryGraphConstantNode):
                data["data"]["label"] = f"{node.constant.value} (const)"
            elif isinstance(node, QueryGraphGeneratedNode):
                vcolumn = None
                for column in wqg.columns:
                    if column.node.id_ == node.id_:
                        vcolumn = column
                for column in wqg.vcolumns:
                    if column.node.id_ == node.id_:
                        vcolumn = column
                data["data"]["label"] = f"{vcolumn} (syn.)"
            nodes.append(data)

        # networkx Graph used for layouting.
        G = nx.Graph()
        for node in nodes:
            G.add_node(node["id"])

        # TODO(jlscheerer) Display all the edges (aggregates and filters).
        edges = []
        for i, edge in enumerate(wqg.edges + wqg.aggregates):
            data = {"id": f"e{i}", "source": str(edge.source.id_.value), "target": str(edge.target.id_.value)}
            if isinstance(edge, QueryGraphPropertyEdge):
                if isinstance(edge.property, QueryGraphPropertyNode):
                    data["label"] = f"'{edge.property.property}'"
                elif isinstance(edge.property, QueryGraphPropertyConstantNode):
                    data["label"] = f"'{edge.property.constant.value}'"
            elif isinstance(edge, QueryGraphFilter):
                pass
            elif isinstance(edge, QueryGraphAggregate):
                data["label"] = edge.type_.name
            G.add_edge(data["source"], data["target"])
            edges.append(data)

        positions = nx.kamada_kawai_layout(G)
        for node in nodes:
            position = positions[node["id"]]
            node["position"] = {
                "x": position[0] * 200,
                "y": position[1] * 200
            }

        # NOTE Hacky way to fix the layout in ReactFlow
        #      The higher element is considered the source, the lower one the target.
        for edge in edges:
            source, target = edge["source"], edge["target"]
            if positions[source][1] > positions[target][1]:
                edge["source"] = target
                edge["target"] = source

        # TODO(jlscheerer) Visualize columns.
        graph = {
            "nodes": nodes,
            "edges": edges
        }

        grouped_query_results = [{"derivation": key, "results": list(group)} for key, group in itertools.groupby(query_results, key=lambda x: x["derivation"])]
        search_uuid_results[id_] = {
            "uuid": id_,
            "query": query,
            "QirK": escape_ansi(pq.canonical()),
            "graph": graph,
            "SPARQL": qs.value,
            "SQL": sql.value,
            "results": grouped_query_results,
            "time": f"{seconds:.02f} seconds"
        }
    finally:
        search_uuid_status[id_] = "Done"


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/search/query")
@cross_origin()
def search_query():
    query = request.args.get("query", type=str)
    id_ = str(uuid.uuid1())
    timestamp = time.time_ns()

    job = {"uuid": id_, "query": query, "timestamp": timestamp}
    thread = threading.Thread(target=handle_search_request, args=[job], daemon=True)
    thread.start()

    return job


@app.route("/search/status")
@cross_origin()
def search_status():
    id_ = request.args.get("uuid", type=str)
    return {"uuid": id_, "status": search_uuid_status[id_]}

@app.route("/search/results")
@cross_origin()
def search_results():
    id_ = request.args.get("uuid", type=str)
    assert search_uuid_status[id_] == "Done"

    results = search_uuid_results[id_]
    return {"uuid": id_, "query": results["query"], "QirK": results["QirK"], "graph": results["graph"],
            "SQL": results["SQL"], "SPARQL": results["SPARQL"], "results": results["results"], "time": results["time"]}

@app.route("/entity")
@cross_origin()
def entity():
    id_ = request.args.get("id", type=str)
    
    db = Database()
    info = db.fetchall(f"""
    SELECT l.value, d.value
    FROM labels_en l, descriptions_en d
    WHERE l.id = '{id_}' AND d.id = '{id_}'
    """)
    assert len(info) == 1
    info = info[0]

    image = db.fetchall(f"""
    SELECT c.datavalue_string, q.datavalue_string
    FROM claims_inv c, qualifiers q
    WHERE c.entity_id = '{id_}' AND c.property = 'P18'
      AND q.claim_id = c.id AND q.qualifier_property = 'P2096'
    ORDER BY SUBSTRING(id FROM '[0-9]+$')::int LIMIT 1
    """)
    assert len(image) <= 1
    if len(image) == 1:
        caption = image[0][1]
        image = retrieve_wiki_image(image[0][0])
    else:
        caption = None
        image = None

    # TODO(jlscheerer) Display all supported information here.
    properties = db.fetchall(f"""
    SELECT c.property, lp.value, c.datavalue_entity, lde.value
    FROM claims_inv c, labels_en lp, labels_en lde
    WHERE c.entity_id = '{id_}'
      AND datavalue_type = 'wikibase-entityid'
      AND lp.id = c.property
      AND lde.id = c.datavalue_entity
    ORDER BY SUBSTRING(c.property FROM '[0-9]+$')::int
    """)

    properties = itertools.groupby(properties, key=lambda x: (x[0], x[1]))
    vproperties = [{
            "property": {
                "name": property[0][1],
                "id": property[0][0],
                "link": "link"
            },
            "data": [
                {
                    "name": element[3],
                    "id": element[2],
                    "link": "link"
                }
                for element in property[1]
            ]
        } for property in properties]

    return {"id": id_, "label": info[0], "description": info[1], 
            "properties": vproperties, "caption": caption, "image": image}

@app.route("/property")
@cross_origin()
def property():
    id_ = request.args.get("id", type=str)
    return {"id": id_}
