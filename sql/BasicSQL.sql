-- Introduction to SQL Basics

-- SQL (Structured Query Language) is a language for 
-- managing and querying relational databases.

-- SELECT: Lists the columns you want to retrieve 
-- FROM: Specifies the table to query.
-- WHERE: Filters rows based on conditions.
-- HAVING: Filters groups based on a condition, often used with aggregate functions.
-- GROUP BY: Groups rows sharing a property, often for aggregate functions.
-- ORDER BY: Sorts the result by one or more columns.

-- Example Query
SELECT column_name1, column_name2 
FROM table_name 
WHERE condition 
ORDER BY column_name1;

-- Key OMOP CDM Tables

-- Person: Contains patient demographics.
-- Condition_Occurrence: Details patient diagnoses.
-- Drug_Exposure: Records medications prescribed or administered.
-- Observation: Stores additional patient observations.
-- Visit_Occurrence: Information about healthcare visits.


-- 1. Retrieving Patient Demographics
-- This query fetches basic patient information, like age and gender.
SELECT person_id, gender_concept_id, year_of_birth, race_concept_id 
FROM person
WHERE year_of_birth >= 1950;

-- Note: Concept IDs reference standardized values in OMOP.
-- For instance, gender_concept_id refers to a code for a patient's gender.

-- 2. Identifying Diagnosed Conditions
-- To find all patients diagnosed with a specific condition (e.g., Hypertension):
SELECT person_id, condition_concept_id, condition_start_date 
FROM condition_occurrence
WHERE condition_concept_id = 31967; -- Hypertension

-- 3. Finding Patients on Specific Medication
-- This query identifies patients prescribed a certain drug (e.g., insulin).
SELECT person_id, drug_concept_id, drug_exposure_start_date, drug_type_concept_id
FROM drug_exposure
WHERE drug_concept_id = 19009384; -- insulin isophane

-- 4. Aggregating Data: Count Diagnoses by Year
-- Count the number of Anemia diagnoses per year:
SELECT EXTRACT(YEAR FROM condition_start_date) AS year, COUNT(*) AS diagnosis_count
FROM condition_occurrence
WHERE condition_concept_id = 439777
GROUP BY year
ORDER BY year;

-- 5. Analyzing Visit Data
-- To find the number of hospital visits each patient has had:
SELECT person_id, COUNT(*) AS visit_count
FROM visit_occurrence
GROUP BY person_id
ORDER BY visit_count DESC;



-- JOIN: Combines rows from different tables based on a shared column

-- 6. Linking Tables with JOINs
-- Query to find the drug concept name linked to the drug concept id
SELECT DISTINCT de.person_id, de.drug_concept_id, c.concept_name AS drug_name
FROM drug_exposure de
JOIN concept c ON de.drug_concept_id = c.concept_id
WHERE de.drug_concept_id = 19009384
LIMIT 5;

-- 7. Using Subqueries
-- Find patients who had a drug exposure after being diagnosed with Hypertension.
SELECT person_id, drug_concept_id, drug_exposure_start_date
FROM drug_exposure
WHERE person_id IN (
    SELECT person_id
    FROM condition_occurrence
    WHERE condition_concept_id = 31967
)

-- 8) Common Table Expressions(CTE)
-- Find drugs prescribed to patients diagnosed with both Hypertension and Diabetes.
WITH hypertension_patients AS (
    SELECT person_id
    FROM condition_occurrence
    WHERE condition_concept_id = 31967 -- Hypertension concept ID
),
diabetes_patients AS (
    SELECT person_id
    FROM condition_occurrence
    WHERE condition_concept_id = 201826 -- Diabetes concept ID
),
hypertension_diabetes_patients AS (
    SELECT person_id
    FROM hypertension_patients
    INTERSECT
    SELECT person_id
    FROM diabetes_patients
)
SELECT DISTINCT hd.person_id, de.drug_concept_id, c.concept_name AS drug_name
FROM hypertension_diabetes_patients hd
JOIN drug_exposure de ON hd.person_id = de.person_id
JOIN concept c ON de.drug_concept_id = c.concept_id;







