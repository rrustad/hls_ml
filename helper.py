# Databricks notebook source
import requests,json, time

# COMMAND ----------

url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None) 
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
_token = {'Authorization': 'Bearer {0}'.format(token)}

# COMMAND ----------

job_id = 1092680156638713
data = {"job_id":job_id}
r = requests.post(url + '/api/2.1/jobs/run-now', headers=_token, data=json.dumps(data))
r

# COMMAND ----------


'runs' in requests.get(url + f'/api/2.1/jobs/runs/list?job_id={job_id}&active_only=true', headers=_token).json().keys()

# COMMAND ----------

r = requests.get(url + f'/api/2.1/jobs/runs/list?job_id={job_id}&active_only=true', headers=_token)
for i in range(30):
  while 'runs' in requests.get(url + f'/api/2.1/jobs/runs/list?job_id={job_id}&active_only=true', headers=_token).json().keys():
    time.sleep(10)
    print('sleep')

  requests.post(url + '/api/2.1/jobs/run-now', headers=_token, data=json.dumps(data))
  print('trigger')



# COMMAND ----------

data = {}

# COMMAND ----------

requests.post()
