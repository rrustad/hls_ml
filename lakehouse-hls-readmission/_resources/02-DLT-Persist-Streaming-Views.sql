-- Databricks notebook source
-- MAGIC %md 
-- MAGIC ## Persist DLT streaming view
-- MAGIC To easily support DLT / UC / ML during the preview with all cluster types, we temporary recopy the final DLT view to another UC table 

-- COMMAND ----------

CREATE OR REPLACE TABLE dbdemos.hls_ml.drug_exposure_ml AS SELECT * FROM dbdemos.hls_ml.drug_exposure;
CREATE OR REPLACE TABLE dbdemos.hls_ml.person_ml AS SELECT * FROM dbdemos.hls_ml.person;
CREATE OR REPLACE TABLE dbdemos.hls_ml.patients_ml AS SELECT * FROM dbdemos.hls_ml.patients;
CREATE OR REPLACE TABLE dbdemos.hls_ml.encounters_ml AS SELECT * FROM dbdemos.hls_ml.encounters;
CREATE OR REPLACE TABLE dbdemos.hls_ml.condition_occurrence_ml AS SELECT * FROM dbdemos.hls_ml.condition_occurrence;
CREATE OR REPLACE TABLE dbdemos.hls_ml.conditions_ml AS SELECT * FROM dbdemos.hls_ml.conditions;

-- COMMAND ----------

-- DROP TABLE dbdemos.hls_ml.drug_exposure_ml;
-- DROP TABLE dbdemos.hls_ml.person_ml;
-- DROP TABLE dbdemos.hls_ml.patients_ml;
-- DROP TABLE dbdemos.hls_ml.encounters_ml;
-- DROP TABLE dbdemos.hls_ml.condition_occurrence_ml;
-- DROP TABLE dbdemos.hls_ml.conditions_ml;

-- COMMAND ----------


