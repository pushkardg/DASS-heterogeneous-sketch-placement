from google.cloud import bigquery
import csv, os

TABLE="bigquery-public-data.chicago_taxi_trips.taxi_trips"

def extract(project,output_csv,metric="trip_total",max_rows=2000000,
            start_date=None,end_date=None,sample=None):
    allowed={"trip_total","trip_miles","trip_seconds"}
    if metric not in allowed:
        raise ValueError(f"metric must be one of {sorted(allowed)}")
    client=bigquery.Client(project=project)
    filters=[
        "trip_start_timestamp IS NOT NULL",
        f"{metric} IS NOT NULL",
        f"{metric} >= 0"
    ]
    if start_date:
        filters.append(f"DATE(trip_start_timestamp) >= '{start_date}'")
    if end_date:
        filters.append(f"DATE(trip_start_timestamp) < '{end_date}'")
    sample_clause=f"TABLESAMPLE SYSTEM ({float(sample)} PERCENT)" if sample else ""
    query=f'''
      SELECT trip_start_timestamp, {metric} AS value
      FROM `{TABLE}` {sample_clause}
      WHERE {' AND '.join(filters)}
      ORDER BY trip_start_timestamp
      LIMIT {int(max_rows)}
    '''
    os.makedirs(os.path.dirname(output_csv) or ".",exist_ok=True)
    with open(output_csv,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["timestamp","value"])
        for row in client.query(query).result(page_size=10000):
            w.writerow([row.trip_start_timestamp.isoformat(),float(row.value)])
    return output_csv
