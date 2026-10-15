"""Flight arrival-delay regression with a chronological holdout."""
import argparse,json
from pathlib import Path
import pandas as pd,numpy as np,joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--input",required=True);p.add_argument("--date",default="FL_DATE");p.add_argument("--target",default="ARR_DELAY");p.add_argument("--out",default="outputs");a=p.parse_args()
 d=pd.read_csv(a.input);d[a.date]=pd.to_datetime(d[a.date],errors="coerce");d[a.target]=pd.to_numeric(d[a.target],errors="coerce");d=d.dropna(subset=[a.date,a.target]).sort_values(a.date).copy()
 # Remove post-arrival and outcome-derived fields to avoid leakage.
 drop=[c for c in d if c.upper() in {"ARR_DELAY","ARR_DELAY_NEW","ARR_DEL15","ARR_TIME","ACTUAL_ELAPSED_TIME","AIR_TIME","WHEELS_ON","WHEELS_OFF"} and c!=a.target]
 d=d.drop(columns=drop,errors="ignore");d["month"]=d[a.date].dt.month;d["weekday"]=d[a.date].dt.dayofweek;d["dayofyear"]=d[a.date].dt.dayofyear
 X=d.drop(columns=[a.target,a.date]);y=d[a.target].clip(-60,240);cut=int(len(d)*.8);Xtr,Xte=X.iloc[:cut],X.iloc[cut:];ytr,yte=y.iloc[:cut],y.iloc[cut:]
 num=X.select_dtypes(include="number").columns;cat=X.columns.difference(num)
 prep=ColumnTransformer([("num",SimpleImputer(strategy="median",add_indicator=True),num),("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(handle_unknown="ignore",max_categories=100))]),cat)])
 # Dense matrix is needed by this estimator; cap input file to a route/time slice for workstation-scale use.
 m=Pipeline([("features",prep),("model",RandomForestRegressor(n_estimators=160,min_samples_leaf=3,n_jobs=-1,random_state=42))]);m.fit(Xtr,ytr);pred=m.predict(Xte)
 metrics={"train_rows":len(Xtr),"test_rows":len(Xte),"mae_minutes":float(mean_absolute_error(yte,pred)),"rmse_minutes":float(mean_squared_error(yte,pred)**.5),"split":"last 20% by flight date","target_clipped_minutes":[-60,240]}
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/"metrics.json").write_text(json.dumps(metrics,indent=2));pd.DataFrame({"actual_delay":yte,"predicted_delay":pred}).to_csv(out/"holdout_predictions.csv",index=False);joblib.dump({"model":m,"features":X.columns.tolist()},out/"model.joblib");print(json.dumps(metrics,indent=2))
if __name__=="__main__":main()
