from .AbstractQueryGraph import AbstractQueryGraph, QueryStatistics

def construct_pid_list(pid_list):
    pid_list_single_quote = [f"'{x}'" for x in pid_list]
    return ", ".join(pid_list_single_quote)


def construct_sqlir_from_aqg(aqg: AbstractQueryGraph):
    edgelist = [(k, v) for k, v in aqg.edges.items()]

    # Construct mapping, from each var to edge id & position
    var2edges = dict()
    for i, (k, _) in enumerate(edgelist):
        sid, oid = k
        var2edges[sid] = var2edges.get(sid, []) + [(i, 1)]
        var2edges[oid] = var2edges.get(oid, []) + [(i, 0)]

    return SQL_IR(aqg, edgelist, var2edges)


def strip_qid_if_needed(s):
    if len(s) > 6 and (s[-5:] == "(pid)" or s[-5:] == "(qid)"):
        return s[:-6]
    else:
        return s


class SQL_IR:
    def __init__(self, graph, edgelist, var2edges):
        self.graph = graph
        self.edgelist = edgelist
        self.var2edges = var2edges

    def get_n_heads(self):
        return len(self.graph.head_vars_ids)

    def construct_from(self):
        if self.graph.is_templated():
            claims_relname = "claims_5m"
        else:
            # claims_relname = 'claims_inv_dt_is_item'
            claims_relname = "claims_5m_inv"
        claims = [f"{claims_relname} c{i}" for i in range(0, len(self.edgelist))]
        return ", ".join(claims)

    def get_all_sql_address_for_var(self, varid):
        adrs = []
        # Check if current varid is going to be a filter.
        # If that's the case, there must be 1 edge (we make it non-joinable for now.)
        # Then, this edge must be going out of something, and endpoint must be "datavalue_string"
        object_type = "datavalue_entity"
        if varid in self.graph.filter_var_ids:
            assert len(self.var2edges[varid]) == 1
            # Todo: other object types?
            object_type = "datavalue_string"
        all_addrs_raw = self.var2edges[varid]
        for edge_id, is_subj in all_addrs_raw:
            so_type = "entity_id" if is_subj else object_type
            adrs.append(f"c{edge_id}.{so_type}")
        return adrs

    def construct_select(self):
        heads = []
        col_names = []
        if self.graph.is_templated():
            for i, _ in enumerate(self.graph.edges):
                heads.append(f"c{i}.property")
            return "DISTINCT " + ", ".join(heads), []
        elif self.graph.is_executable():
            # Select all properties (predicates)
            for i, (_, edge) in enumerate(self.graph.edges.items()):
                col_name = f"{edge.get_up()} ({i}) (pid)"
                heads.append(f'c{i}.property AS "{col_name}"')
                col_names.append(col_name)
            # Select all anchors
            num_anchors = 0
            for node in self.graph.nodes:
                if not node.is_free:
                    node_addr = self.get_all_sql_address_for_var(node.id)[0]
                    col_name = f"{node.get_str()} (qid)"
                    heads.append(f'{node_addr} AS "{col_name}"')
                    col_names.append(col_name)
                    num_anchors += 1
            # Select all head free vars.
            for i, hv_id in enumerate(self.graph.head_vars_ids):
                adrs = self.get_all_sql_address_for_var(hv_id)
                # We are free to use any address for this var when it's a head.
                col_name = self.graph.nodes[hv_id].get_str()
                heads.append(f"{adrs[0]} AS {col_name}")
                col_names.append(col_name)
            nums_and_cols_stats = [
                len(self.graph.edges),
                num_anchors,
                len(self.graph.head_vars_ids),
                col_names,
            ]
            return ", ".join(heads), nums_and_cols_stats
        else:
            print("construct_select(): graph neither is_templated nor is_executable")
            return ""

    def construct_where_join_condition(self, node, where_conds, all_adrs):
        join_cond = []
        all_adrs += [all_adrs[0]]
        cond = [f"{all_adrs[i]} = {all_adrs[i+1]}" for i in range(len(all_adrs) - 1)]
        where_conds.append(" AND ".join(cond))

    def construct_where_anchor_node_templated(self, node, where_conds):
        node_addr = self.get_all_sql_address_for_var(node.id)[0]
        qid = self.graph.anchors_to_wiki[node.id]
        where_conds.append(f"{node_addr} = '{qid}'")

    def construct_where_anchor_node_executable(self, node, where_conds):
        node_addr = self.get_all_sql_address_for_var(node.id)[0]
        qids = self.graph.anchors_to_wiki[node.id]
        qids_list = construct_pid_list(qids)
        where_conds.append(f"{node_addr} IN ({qids_list})")

    def add_edge_filter_for_templates(self, where_conds):
        for i, _ in enumerate(self.graph.edges):
            where_conds.append(
                f"c{i}.property in (select pid from pids_of_interest_entities)"
            )

    def construct_where(self):
        where_conds = []
        if self.graph.is_templated():
            for node in self.graph.nodes:
                all_adrs = self.get_all_sql_address_for_var(node.id)
                if len(all_adrs) >= 2:
                    self.construct_where_join_condition(node, where_conds, all_adrs)
                if not node.is_free:
                    self.construct_where_anchor_node_templated(node, where_conds)
            self.add_edge_filter_for_templates(where_conds)
            return " AND ".join(where_conds)
        elif self.graph.is_executable():
            for i, (_, edge) in enumerate(self.edgelist):
                where_conds.append(
                    f"c{i}.property IN ({construct_pid_list(edge.get_pids())})"
                )
            for node in self.graph.nodes:
                all_adrs = self.get_all_sql_address_for_var(node.id)
                if len(all_adrs) >= 2:
                    self.construct_where_join_condition(node, where_conds, all_adrs)
                if not node.is_free:
                    self.construct_where_anchor_node_executable(node, where_conds)
            # Add filtering conditions.
            for filter_ in self.graph.filters:
                filter_id = self.graph.var2ids[filter_[0]]
                addr = self.get_all_sql_address_for_var(filter_id)[0]
                where_conds.append(
                    f"substring({addr}, '\\d+')::integer {filter_[1]} {filter_[2]}"
                )
            return " AND ".join(where_conds)
        else:
            print("construct_where(): graph neither is_templated nor is_executable")
            return ""

    def sqlize(self, query_stats=QueryStatistics(), with_all_labels=False):
        SELECT, nums_and_cols_stats = self.construct_select()
        if not self.graph.is_templated():
            query_stats.add_nums_and_cols(*nums_and_cols_stats)
        FROM = self.construct_from()
        WHERE = self.construct_where()
        orig_query = f"""SELECT {SELECT} FROM {FROM} WHERE {WHERE}"""
        if not with_all_labels:
            return f"{orig_query};"
        else:
            return self.join_labels(orig_query, query_stats.col_names)

    def join_labels(self, orig_query, col_names):
        selections = ", ".join(
            [
                f'oq."{cname}", l{i}.value AS "{strip_qid_if_needed(cname)} (label)"'
                for (i, cname) in enumerate(col_names)
            ]
        )
        ljoins = " ".join(
            [
                f'LEFT JOIN labels_en l{i} ON l{i}.id = oq."{cname}"'
                for (i, cname) in enumerate(col_names)
            ]
        )
        full_query = f"""WITH orig_query AS ({orig_query}) SELECT {selections} FROM orig_query oq {ljoins};"""
        return full_query


def wqg2sql(wqg, query_stats):
    sql_ir = construct_sqlir_from_aqg(wqg)
    should_join_labels = True
    sql_str = sql_ir.sqlize(query_stats, should_join_labels)
    return sql_str, query_stats
