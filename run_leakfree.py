#!/usr/bin/env python3
import os, sys, time, subprocess
os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-17-openjdk-amd64'
os.environ['PATH'] = os.environ['JAVA_HOME'] + '/bin:' + os.environ['PATH']

import h2o
from h2o.automl import H2OAutoML

root = '/media/datos/WORKSPACE/DOCTORADO/paper_ugr_modo_automl_insolvency'

for h in [1,3,5]:
    print(f'=== STARTING H{h} ===')
    try:
        h2o.cluster().shutdown(prompt=False)
        time.sleep(3)
    except:
        pass
    # kill any h2o
    subprocess.run('pkill -9 -f h2o.jar 2>&1; sleep 3', shell=True)
    outdir = os.path.join(root, 'runs', f'leakfree_h{h}')
    os.makedirs(outdir, exist_ok=True)
    os.chdir(outdir)
    h2o.init(nthreads=-1, max_mem_size="16G")
    name = f"10_to_predict_{h}_leakfree"
    train_path = os.path.join(root, 'data', 'final', f'train_balanced_leakfree_h{h}.csv')
    test_path = os.path.join(root, 'data', 'final', f'test_unbalanced_leakfree_h{h}.csv')
    train = h2o.import_file(train_path)
    test = h2o.import_file(test_path)
    y = "Y"
    x = list(train.columns)
    x.remove(y)
    for c in ["CIF", "window_id", "ref_year", "horizon"]:
        if c in x:
            x.remove(c)
    train[y] = train[y].asfactor()
    test[y] = test[y].asfactor()
    aml = H2OAutoML(max_models=20, seed=1234, nfolds=5, sort_metric="auc_pr")
    aml.train(x=x, y=y, training_frame=train)
    lb = aml.leaderboard
    df = lb.as_data_frame()
    df.to_csv(os.path.join(outdir, f'{name}_leaderboard.csv'), index=False)
    for model_id in df["model_id"].tolist():
        try:
            m = h2o.get_model(model_id)
            h2o.save_model(model=m, path=outdir, force=True)
        except Exception as e:
            print(e)
    try:
        aml.leader.download_mojo(path=outdir)
    except Exception as e:
        print(e)
    try:
        h2o.cluster().shutdown(prompt=False)
    except:
        pass
    time.sleep(10)
    print(f'=== FINISHED H{h} ===')
