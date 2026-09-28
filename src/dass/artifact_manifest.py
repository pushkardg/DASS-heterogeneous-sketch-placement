import argparse,csv,hashlib,json,os
import pandas as pd

METRIC_PATHS={
    'trip_total':'trip_total',
    'trip_miles':'trip_miles',
    'trip_seconds':'trip_seconds',
}

def _sha256(path,chunk_size=1024*1024):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            chunk=f.read(chunk_size)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()

def _count_csv_rows(path):
    # stream.csv contains one record per line plus a header. csv.reader keeps
    # this robust to quoted values even though the current files are simple.
    with open(path,newline='') as f:
        r=csv.reader(f)
        try: next(r)
        except StopIteration: return 0
        return sum(1 for _ in r)

def build_dataset_manifest(config_path,output_csv):
    with open(config_path) as f:
        config=json.load(f)
    rows=[]
    for item in config:
        replicate=item['name']
        for metric,key in METRIC_PATHS.items():
            path=item[key]
            exists=os.path.isfile(path)
            rows.append({
                'replicate':replicate,
                'metric':metric,
                'path':path,
                'exists':exists,
                'row_count':_count_csv_rows(path) if exists else None,
                'file_bytes':os.path.getsize(path) if exists else None,
                'sha256':_sha256(path) if exists else None,
            })
    df=pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv) or '.',exist_ok=True)
    df.to_csv(output_csv,index=False)
    summary={
        'files':int(len(df)),
        'files_present':int(df['exists'].sum()),
        'total_rows':int(df['row_count'].dropna().sum()),
        'all_files_present':bool(df['exists'].all()),
    }
    summary_path=os.path.splitext(output_csv)[0]+'_summary.json'
    with open(summary_path,'w') as f: json.dump(summary,f,indent=2)
    return df,summary

def main():
    p=argparse.ArgumentParser(description='Build exact row-count/checksum manifest for DASS publication inputs')
    p.add_argument('--config',default='configs/chicago_replicates.json')
    p.add_argument('--output',default='results/chicago-v10/dataset_manifest.csv')
    args=p.parse_args()
    _,summary=build_dataset_manifest(args.config,args.output)
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
