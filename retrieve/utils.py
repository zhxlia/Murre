import json

from copy import deepcopy
from typing import List, Dict, Any


def pack_table(db: Dict[str, Any], if_orginal: bool) -> str:
    # Extract information from db
    if not if_orginal:
        column_names = db["column_names"]
    else:
        column_names = db["column_names_original"]
    column_types = db["column_types"]
    primary_keys = db["primary_keys"]
    foreign_keys = db["foreign_keys"]
    if not if_orginal:
        table_names = db["table_names"]
    else:
        table_names = db["table_names_original"]

    # Helper function to get table's columns
    def get_table_columns(table_index):
        return [(name, column_types[i]) for i, (idx, name) in enumerate(column_names) if idx == table_index]

    # Helper function to get table's primary keys
    def get_table_primary_keys(table_index):
        return [name for i, (idx, name) in enumerate(column_names) if idx == table_index and i in primary_keys]

    # Helper function to get table's foreign keys
    def get_table_foreign_keys(table_index):
        fk_list = []
        for (from_idx, to_idx) in foreign_keys:
            if from_idx in [i for i, (idx, name) in enumerate(column_names) if idx == table_index]:
                from_column = column_names[from_idx][1]
                to_table_index = [idx for i, (idx, name) in enumerate(
                    column_names) if i == to_idx][0]
                to_table = table_names[to_table_index]
                to_column = column_names[to_idx][1]
                fk_list.append((from_column, to_table, to_column))
        return fk_list

    # Generate SQL statements
    sql_statements = []
    for i, table_name in enumerate(table_names):
        # Columns
        columns_sql = []
        for column_name, column_type in get_table_columns(i):
            # Map general types to SQL types
            if column_type == "text":
                sql_type = "text"
            elif column_type == "number":
                sql_type = "int"  # This can be adjusted based on further specifications
            else:
                sql_type = column_type
            columns_sql.append(f'{column_name} {sql_type}')

        # Primary keys
        pk_sql = []
        for pk in get_table_primary_keys(i):
            pk_sql.append(f'{pk}')

        # Foreign keys
        fk_sql = []
        for from_column, to_table, to_column in get_table_foreign_keys(i):
            fk_sql.append(
                f'FOREIGN KEY ({from_column}) REFERENCES {to_table}({to_column})')

        # Combine all
        table_sql = f'CREATE TABLE {table_name} (\n'
        table_sql += " ,\n".join(columns_sql +
                                [f'PRIMARY KEY ({", ".join(pk_sql)})'] + fk_sql)
        table_sql += "\n);"
        sql_statements.append(table_sql)

    result = "\n\n".join(sql_statements).lower()
    return result

def re_index(index_map: List[int], ind: int):
    if ind == -1:
        return -1
    else:
        return index_map.index(ind)

def filter_ret_tables_from_db(db_dict: Dict[str, Any], db_id: str, ret_tables_list: List):
    # with open(tables_file, 'r', encoding='utf-8') as f:
    #     origin_tables: List[Dict[str, Any]] = json.load(f)
    #     tables: Dict[str, Dict[str, Any]] = {
    #         x['db_id']: x for x in origin_tables}
    # db = db_dict
    if db_dict is None:
        print(db_id)
    ndb_dict = deepcopy(db_dict)
    # 根据ret_tables_list过滤db中的table
    table_ids = [i for i, x in enumerate(ndb_dict['table_names']) if x in ret_tables_list]
    # print(f"table_ids: {table_ids}")
    column_names_ids= [i for i, x in enumerate(ndb_dict['column_names']) if x[0] in table_ids or x[0] == -1]
    ndb_dict["column_names"]= [x for i, x in enumerate(db_dict['column_names']) if x[0] in table_ids or x[0] == -1]   
    ndb_dict["column_names"]= [[re_index(table_ids, x[0]), x[1]] for i, x in enumerate(ndb_dict['column_names'])]   

    ndb_dict["column_types"]= [x for i, x in enumerate(db_dict['column_types']) if i in column_names_ids]

    ndb_dict["column_names_original"]= [x for i, x in enumerate(db_dict['column_names_original']) if x[0] in table_ids or x[0] == -1]
    ndb_dict["column_names_original"]= [[re_index(table_ids, x[0]), x[1]] for i, x in enumerate(ndb_dict['column_names_original'])] 

    ndb_dict["primary_keys"] = [x for i, x in enumerate(db_dict['primary_keys']) if x in column_names_ids]
    ndb_dict["primary_keys"] = [re_index(column_names_ids, x) for i, x in enumerate(ndb_dict["primary_keys"])]

    ndb_dict["foreign_keys"] = [x for i, x in enumerate(db_dict['foreign_keys']) if x[0] in column_names_ids and x[1] in column_names_ids]
    ndb_dict["foreign_keys"] = [[re_index(column_names_ids, x[0]), re_index(column_names_ids, x[1])] for i, x in enumerate(ndb_dict["foreign_keys"])]
    # ndb_dict["table_names_original"] = []
    ndb_dict["table_names_original"] = [x for i, x in enumerate(db_dict['table_names_original']) if i in table_ids]
    ndb_dict["table_names"] = [x for i, x in enumerate(db_dict['table_names']) if i in table_ids]

    return ndb_dict