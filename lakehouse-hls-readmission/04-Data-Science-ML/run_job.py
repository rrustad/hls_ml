# Databricks notebook source
import requests, time, json

# COMMAND ----------

url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None) 
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
_token = {
            'Authorization': 'Bearer {0}'.format(token)
        }

# COMMAND ----------

count = 0
while count < 30:
  r = requests.get(url+ '/api/2.1/jobs/runs/list/?job_id=255509481591216',headers=_token)
  [job['state']['life_cycle_state'] for job in r.json()['runs']]
  # r.json()
  if 'RUNNING' in [job['state']['life_cycle_state'] for job in r.json()['runs']]:
    print('running')
    time.sleep(10)
  else:
    count += 1
    print('trigger', count)
    data = {
      'job_id':255509481591216,
    }
    r = requests.post(url+ '/api/2.1/jobs/run-now/',headers=_token, data=json.dumps(data))
    print(r.json())
    count += 1
    time.sleep(10)


  


# COMMAND ----------

r = requests.get(url+ '/api/2.1/jobs/runs/list/?job_id=255509481591216',headers=_token)
[job['state']['life_cycle_state'] for job in r.json()['runs']]
# r.json()
if 'RUNNING' in [job['state']['life_cycle_state'] for job in r.json()['runs']]:
  print('running')
  time.sleep(10)
else:
  print('trigger')
  count += 1
  data = {
    'job_id':255509481591216,
  }
  r = requests.post(url+ '/api/2.1/jobs/run-now/',headers=_token, data=json.dumps(data))
  r.json()
  count += 1

# COMMAND ----------


  data = {
    'job_id':255509481591216,
  }
r = requests.post(url+ '/api/2.1/jobs/run-now/',headers=_token, data=json.dumps(data))
r.json()

# COMMAND ----------


