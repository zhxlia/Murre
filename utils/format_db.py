import os
import sqlite3

from typing import Dict, List, Any

DATABASE_TEMPLATE = """
CREATE TABLE {table_name} (
    {columns},
    {primary_key},
    {foreign_key}
)
/*
Columns in {table_name} and 3 examples in each column:
{values};
*/
""".strip()


def format_db(db: Dict[str, Any], db_path: str, sample_number: int) -> str:
    def sample_values(db_path: str, table_name: str, column_names: List[str], number: int) -> Dict[str, List[str]]:
        """
        Load a database from the given path, sample a number of examples from the specified table,
        and return the data in a dictionary format where the order of keys follows the order of column names.
        This function also handles column names with spaces by enclosing them in double quotes.

        :param db_path: Path to the SQLite database.
        :param table_name: Name of the table to sample from.
        :param column_names: List of column names to include in the output.
        :param number: Number of samples to retrieve.
        :return: A dictionary with keys as column names and values as lists of column data.
        """
        try:
            # Connect to the SQLite database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Prepare the query to fetch the data, enclosing column names in double quotes if they contain spaces
            columns = ", ".join(
                [f'"{name}"' if ' ' in name else name for name in column_names])
            query = f"SELECT {columns} FROM {table_name} ORDER BY RANDOM() LIMIT {number}"

            # Execute the query and fetch the data
            cursor.execute(query)
            rows = cursor.fetchall()

            # Organize the data in the desired format
            data = {column: [] for column in column_names}
            for row in rows:
                for i, column in enumerate(column_names):
                    data[column].append(row[i])

            # Close the database connection
            conn.close()

            return data

        except Exception as e:
            return {}

    db_path = os.path.join(db_path, db["db_id"], f"{db['db_id']}.sqlite")
    databases: List[str] = []
    for table_idx, table_name in enumerate(db['table_names_original']):
        columns: List[str] = [c[1] + ' ' + t for c, t in zip(
            db['column_names_original'], db['column_types']) if c[0] == table_idx]

        primary_keys: List[str] = [
            f"PRIMARY KEY ({db['column_names_original'][p][1]})" for p in db['primary_keys'] if db['column_names_original'][p][0] == table_idx]

        foreign_keys: List[str] = []
        for f in db['foreign_keys']:
            if db['column_names'][f[1]][0] == table_idx:
                f[0], f[1] = f[1], f[0]
            if db['column_names'][f[0]][0] == table_idx:
                column_0 = db['column_names_original'][f[0]]
                column_1 = db['column_names_original'][f[1]]
                foreign_keys.append(
                    f"FOREIGN KEY ({column_0[1]}) REFERENCES {db['table_names_original'][column_1[0]]}({column_1[1]})")

        corresponding_values = sample_values(db_path, table_name, [
                                             c[1] for c in db['column_names_original'] if c[0] == table_idx], sample_number)
        corresponding_values = {k: [f'"{x}"' if isinstance(x, str) else str(
            x) for x in v] for k, v in corresponding_values.items()}
        values: List[str] = [
            f"{k}: {', '.join(v)}" for k, v in corresponding_values.items()]

        databases.append(DATABASE_TEMPLATE.format(
            table_name=table_name,
            columns=',\n    '.join(columns),
            primary_key=',\n    '.join(primary_keys),
            foreign_key=',\n    '.join(foreign_keys),
            values=';\n'.join(values)
        ))

    print('\n\n'.join(databases))
    return '\n\n'.join(databases)