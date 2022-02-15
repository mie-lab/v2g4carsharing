import psycopg2
from sqlalchemy import create_engine
import json
import pandas as pd
import os

from import_utils import read_geodataframe


def write_to_postgis(local_path, path_credentials):
    # load credentials
    with open(path_credentials, "r") as infile:
        db_credentials = json.load(infile)

    def get_con():
        return psycopg2.connect(**db_credentials)

    engine = create_engine('postgresql+psycopg2://', creator=get_con)

    # 1) POSTGIS
    df_names = ["user", "station"]
    index_names = ["person_no", "station_no"]
    for load_name, index_name in zip(df_names, index_names):
        df = read_geodataframe(os.path.join(local_path, f"{load_name}.csv")
                               ).set_index(index_name)
        df.to_postgis(
            load_name,
            engine,
            schema="mobility",
            index=True,
            index_label=index_name,
            chunksize=10000,
            dtype=None
        )

    # 2) NORMAL POSTGRESQL
    df_names = ["vehicle", "all_reservation", "vehicle_to_base", "relocation"]
    index_names = ["vehicle_no", "reservation_no", "v2b_no", "relocation_no"]
    for load_name, index_name in zip(df_names, index_names):
        df = pd.read_csv(
            os.path.join(local_path, f"{load_name}.csv", index_col=index_name)
        )
        df.to_sql(
            load_name,
            engine,
            schema="mobility",
            index=True,
            index_label=index_name,
            if_exists='replace',
            chunksize=10000
        )


if __name__ == "__main__":
    path = "data"
    db_login_path = "../../goeco_login.json"
    write_to_postgis(path, db_login_path)
