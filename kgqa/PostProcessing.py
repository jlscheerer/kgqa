from kgqa.Database import Database


def run_and_rank(sql_query, query_stats, wqg):
    db = Database()
    results, column_names = db.fetchall(sql_query, return_column_names=True)
    return results, column_names
