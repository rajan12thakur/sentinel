SENTINEL — Project Engineering Playbook

Purpose

This document records the decisions, architecture, setup process, commands, errors, fixes, PostgreSQL steps, Django steps, and Git workflow used for SENTINEL — Intelligent Predictive Maintenance & Failure Intelligence Platform.

Use it as a reusable checklist when starting future software/ML projects.

1. Project Overview

SENTINEL is an industrial predictive-maintenance platform designed to use machine sensor/time-series data to:

Estimate machine failure risk.

Predict Remaining Useful Life (RUL).

Explain important factors behind predictions.

Expose predictions through a production-style API.

Persist machine/prediction metadata in PostgreSQL.

Provide a web dashboard.

The project is intended to demonstrate Python, Django/DRF, PostgreSQL, SQL/data handling, classical ML, PyTorch/deep learning, time-series modeling, leakage prevention, class-imbalance handling, evaluation, explainability, ML experiment tracking, API development, testing, and production-oriented ML engineering.

Principle: do not add technology just to make the repository look impressive. Each component should solve a real requirement.

2. Business Requirements

Business problem

Industrial equipment can degrade before failure. Unexpected failures can cause downtime, maintenance cost, equipment damage, and operational disruption.

Business objective

Convert historical machine sensor data into actionable predictions:

Sensor Data
    ↓
Data Processing
    ↓
Feature Engineering
    ↓
ML / Deep Learning
    ↓
Failure Risk + RUL
    ↓
API
    ↓
Dashboard / Applications

Functional requirements

FR-01 — Machine data ingestion

Load/receive machine sensor data.

FR-02 — Data preprocessing

Validate data, handle missing values where appropriate, create time-series features, and prevent future information from leaking into training.

FR-03 — Failure-risk prediction

Estimate failure probability/risk.

Example:

{
  "machine_id": "FD001_001",
  "risk_level": "HIGH",
  "failure_probability": 0.87
}

The actual thresholds should be justified by experiments/business assumptions.

FR-04 — RUL prediction

Estimate remaining useful life.

{
  "machine_id": "FD001_001",
  "predicted_rul": 24
}

FR-05 — Explainability

Provide information about important sensors/features contributing to predictions.

FR-06 — API

Expose prediction functionality through REST endpoints.

FR-07 — Persistence

Store machine, prediction, and model metadata in PostgreSQL.

FR-08 — Dashboard

Provide a simple Django UI for system/machine information.

3. Non-Functional Requirements

The system should be:

reproducible

testable

modular

maintainable

version-controlled

reasonably scalable

observable

suitable for later containerization

4. MVP Scope

Included

Backend

Django

Django REST Framework

PostgreSQL

environment-based configuration

health endpoint

basic dashboard

automated tests

ML

C-MAPSS ingestion

EDA

leakage investigation

leakage-safe splitting

feature engineering

classical ML baseline

class-imbalance analysis

custom PyTorch model

at least one sequence model

proper evaluation

error analysis

Engineering

Git branches

GitHub Issues

Pull Requests

tests

documentation

clean commits

5. Explicitly Out of Scope for Initial MVP

These should not block the first working version:

Kubernetes

cloud deployment

complex CI/CD

distributed training

microservices

large-scale streaming infrastructure

complex feature stores

sophisticated multi-tenant architecture

mobile application

advanced notification systems

production autoscaling

Potential future infrastructure:

Docker

Redis

Celery

Gunicorn

CI/CD

cloud deployment

Rule: infrastructure should be introduced when there is an actual requirement.

6. Technology Decisions

Django

Chosen for a mature Python backend, APIs, and dashboard support.

Django REST Framework

Chosen for REST APIs, serializers, validation, and API testing.

PostgreSQL

Chosen for reliable relational storage of structured machine/prediction metadata.

scikit-learn

For preprocessing and classical ML baselines.

Potential models:

Logistic Regression
Random Forest
XGBoost

PyTorch

For custom deep-learning and sequence models.

Potential progression:

MLP
 ↓
1D CNN
 ↓
LSTM
 ↓
Transformer Encoder (later)

MLflow

To be introduced when meaningful ML experiments begin.

7. Dataset Strategy

Primary proposed dataset:

NASA C-MAPSS turbofan engine degradation/prognostics dataset.

Before modeling:

Understand dataset structure.

Identify machine/unit IDs.

Understand operating cycles.

Inspect sensor behavior.

Check missing/constant features.

Investigate leakage.

Design the split strategy.

Build features.

8. ML Problem Definition

Failure-risk classification

Input:

Machine sensor history

Output:

Failure probability / risk category

Important metrics:

Precision

Recall

F1

PR-AUC

ROC-AUC

confusion matrix

false negatives

Accuracy must not be the only metric when failure classes are imbalanced.

RUL regression

Potential metrics:

MAE

RMSE

R²

The final metric set should reflect the actual modeling/business objective.

9. ML Methodology

Stage 1 — Data understanding

Perform:

schema inspection

descriptive statistics

distributions

correlation analysis

machine-level analysis

cycle/time analysis

Stage 2 — Leakage investigation

Critical questions:

Is the same machine present in train and validation?

Are future cycles used to create historical features?

Are labels using information unavailable at prediction time?

Are scalers/statistics fitted using validation/test data?

Stage 3 — Data splitting

Use a strategy appropriate to the data-generating process, considering:

machine/group-aware splits

temporal splits

train/validation/test separation

The exact strategy must be documented and justified.

Stage 4 — Feature engineering

Potential features:

rolling mean

rolling standard deviation

rolling min/max

deltas

rates of change

slopes

lag features

operating-condition features

Do not automatically keep every feature.

Stage 5 — Classical ML baseline

Suggested:

Logistic Regression
        ↓
Random Forest
        ↓
XGBoost

The baseline establishes a reference point for deep learning.

Stage 6 — Imbalance

Investigate:

class distribution

minority recall

class weights

weighted sampling

threshold selection

focal loss for PyTorch if justified

Stage 7 — Custom PyTorch training

Implement:

Dataset

DataLoader

model

forward pass

loss

optimizer

training loop

validation loop

checkpointing

early stopping

Stage 8 — Sequence modeling

Create temporal sequences and compare:

MLP → 1D CNN → LSTM → Transformer later

Do not assume the most complex model will win.

Stage 9 — Multi-task learning (later)

Potential architecture:

             Shared Encoder
                   │
          ┌────────┴────────┐
          ↓                 ↓
 Failure Head           RUL Head
Classification          Regression

Possible objective:

Total Loss =
classification loss + λ × regression loss

Choose λ experimentally.

10. Model Evaluation Philosophy

Never fabricate metrics.

Correct workflow:

Train
  ↓
Evaluate
  ↓
Compare
  ↓
Analyze errors
  ↓
Select model based on evidence

If XGBoost performs better than LSTM, keep XGBoost and explain why.

If LSTM performs better, keep LSTM.

The project demonstrates engineering judgment, not model complexity.

11. Explainability

For classical models:

SHAP

feature importance

For deep learning:

feature attribution

ablation studies

sensor-level analysis

Question to answer:

Why did the model predict high failure risk?

12. Proposed Architecture

                    ┌─────────────────────┐
                    │   Sensor Dataset    │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Validation &        │
                    │ Preprocessing       │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    └──────────┬──────────┘
                               ↓
             ┌─────────────────┴─────────────────┐
             ↓                                   ↓
    ┌──────────────────┐                ┌──────────────────┐
    │ Classical ML     │                │ Deep Learning    │
    │ Baselines        │                │ Models           │
    └────────┬─────────┘                └────────┬─────────┘
             └────────────────┬──────────────────┘
                              ↓
                    ┌─────────────────────┐
                    │ Evaluation &        │
                    │ Error Analysis      │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Selected Model      │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Django / DRF        │
                    │ Inference API       │
                    └──────────┬──────────┘
                               ↓
                    ┌──────────┴──────────┐
                    ↓                     ↓
             ┌──────────────┐     ┌──────────────┐
             │ PostgreSQL   │     │ Dashboard    │
             └──────────────┘     └──────────────┘

13. Repository Architecture

Current intended structure:

sentinel/
├── config/
├── core/
├── dashboard/
├── machines/
├── predictions/
├── ml_models/
├── .env.example
├── .gitignore
├── docker-compose.yml
├── manage.py
└── requirements.txt

We intentionally changed from multiple requirements files to one:

requirements.txt

Do not create empty modules just for appearance. Introduce functionality through actual issues.

14. Dependency Management

Current requirements:

Django
djangorestframework
psycopg[binary]
python-dotenv
pytest
pytest-django

As ML work starts, add only what is actually needed, potentially:

numpy
pandas
scikit-learn
matplotlib
seaborn
jupyter
torch
torchvision
mlflow

Later if justified:

celery
redis
gunicorn

Avoid running pip freeze into the project too early because the environment can contain unrelated packages.

Once dependencies stabilize, versions can be pinned.

15. Environment Configuration

.env contains local secrets/configuration:

DEBUG=True

SECRET_KEY=change-this-development-secret-key

DB_NAME=sentinel
DB_USER=sentinel_user
DB_PASSWORD=admin
DB_HOST=localhost
DB_PORT=5432

.env.example should contain placeholders, not real credentials:

DEBUG=True
SECRET_KEY=
DB_NAME=sentinel
DB_USER=
DB_PASSWORD=
DB_HOST=localhost
DB_PORT=5432

Never commit .env.

16. Django Configuration

At the top of config/settings.py:

from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

Database:

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.getenv("DB_NAME"),
        "USER": os.getenv("DB_USER"),
        "PASSWORD": os.getenv("DB_PASSWORD"),
        "HOST": os.getenv("DB_HOST"),
        "PORT": os.getenv("DB_PORT"),
    }
}

17. PostgreSQL Setup — Actual Process

PostgreSQL 16 was already installed.

Check Windows service:

Get-Service *postgres*

Result:

Running  postgresql-x64-16

Initially:

psql --version

returned:

psql : The term 'psql' is not recognized...

This did not mean PostgreSQL was missing. It meant psql.exe was not in the Windows PATH.

Find it:

Get-ChildItem "C:\Program Files\PostgreSQL" -Recurse -Filter psql.exe -ErrorAction SilentlyContinue

Found:

C:\Program Files\PostgreSQLin\psql.exe

Run:

& "C:\Program Files\PostgreSQLin\psql.exe" --version

Result:

psql (PostgreSQL) 16.14

Connect:

& "C:\Program Files\PostgreSQLin\psql.exe" -U postgres

18. PostgreSQL Commands Used

Inside psql:

List databases:

\l

List roles:

\du

Query databases:

SELECT datname FROM pg_database;

Query roles:

SELECT rolname FROM pg_roles;

Connect to database:

\c sentinel

Reset unfinished SQL input:



Exit:

\q

19. PostgreSQL Authentication Problem

Existing databases included:

postgres
retail_customer_behavior
datawarehouse
sentinel

But sentinel_user did not exist.

Therefore Django's earlier authentication attempt using:

DB_USER=sentinel_user

failed.

Instead of using the PostgreSQL superuser from Django, we created a dedicated application role.

Create:

CREATE USER sentinel_user WITH PASSWORD 'admin';

Grant database access:

GRANT ALL PRIVILEGES ON DATABASE sentinel TO sentinel_user;

Connect:

\c sentinel

Grant schema permissions required for Django migrations:

GRANT USAGE, CREATE ON SCHEMA public TO sentinel_user;

Verify:

\du

Result included:

sentinel_user

Then exit:

\q

Security note

admin was used for local development during this setup. A real deployment should use a strong generated password, environment/secret management, and separate credentials for environments.

20. Django/PostgreSQL Verification

Run:

python manage.py check

Successful result:

System check identified no issues (0 silenced).

Then:

python manage.py migrate

Django successfully applied:

contenttypes
auth
admin
sessions

This verified:

Django
 ↓
psycopg
 ↓
PostgreSQL 16
 ↓
sentinel database
 ↓
sentinel_user

21. Docker Decision

Docker was attempted and the machine returned:

docker : The term 'docker' is not recognized...

Decision:

Do not make Docker a blocker for Issue #1.

Use the existing local PostgreSQL installation.

Dockerization becomes a future infrastructure issue.

Engineering principle:

Do not let infrastructure that is not required for the MVP block the core application.

22. Health Endpoint

Target:

GET /api/core/health/

Expected:

{
  "status": "ok",
  "service": "sentinel"
}

Example view:

from django.http import JsonResponse

def health_check(request):
    return JsonResponse({
        "status": "ok",
        "service": "sentinel"
    })

Example app URL:

from django.urls import path
from .views import health_check

urlpatterns = [
    path("health/", health_check, name="health-check"),
]

Project URL:

path("api/core/", include("core.urls")),

23. Dashboard

Initial dashboard is intentionally simple.

Purpose:

verify Django templates

verify static files

establish UI foundation

later display machine/prediction information

Target:

http://127.0.0.1:8000/

Do not build a sophisticated dashboard before the prediction pipeline exists.

24. Testing

Basic health test:

from django.test import TestCase

class HealthCheckTest(TestCase):

    def test_health_check(self):
        response = self.client.get("/api/core/health/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

Run:

python manage.py test

Rule:

Every meaningful feature should eventually have automated tests.

25. Useful Django Commands

Create project:

django-admin startproject config .

Create app:

python manage.py startapp core

Check:

python manage.py check

Create migrations:

python manage.py makemigrations

Apply migrations:

python manage.py migrate

Run tests:

python manage.py test

Run server:

python manage.py runserver

Create admin:

python manage.py createsuperuser

26. PowerShell / VS Code Commands

Activate virtual environment:

.\.venv\Scripts\Activate.ps1

Show current directory:

Get-Location

List files:

Get-ChildItem

Create directory:

New-Item directory-name -ItemType Directory

Create file:

New-Item filename.txt

Remove directory:

Remove-Item -Recurse -Force directory-name

Read file:

Get-Content .env

Open project in VS Code:

code .

Find PostgreSQL executable:

Get-ChildItem "C:\Program Files\PostgreSQL" -Recurse -Filter psql.exe -ErrorAction SilentlyContinue

27. Git Strategy

Branch model:

main
 ↑
develop
 ↑
issue / feature branches

main

Production/release branch.

Do not directly develop here.

develop

Integration branch.

Feature/issue branches merge here.

Issue branch

Created from develop.

Current branch:

1-project-initialization-and-django-setup

Correct flow:

Issue Branch
     ↓
git push
     ↓
GitHub Branch
     ↓
Pull Request
     ↓
develop
     ↓
Release Pull Request
     ↓
main

Never send an issue branch directly into main.

28. Git Commands

Status:

git status

Branches:

git branch -vv

History:

git log --oneline --decorate --graph --all

Fetch:

git fetch origin

Pull:

git pull origin develop

Switch:

git checkout develop

Create issue/feature branch:

git checkout -b feature/<issue-number>-<description>

Stage:

git add -A

Review:

git status
git diff --cached

Commit:

git commit -m "type: description"

First push:

git push -u origin <branch-name>

Later pushes:

git push

Delete local branch after merge:

git branch -d <branch-name>

Delete remote branch:

git push origin --delete <branch-name>

29. Commit Strategy

Use small logical commits.

Examples:

chore: simplify dependency management
chore: configure Django application
feat: add health endpoint
feat: add dashboard foundation
test: add health endpoint test

Avoid:

final project
everything
changes
update

A commit should describe one logical change.

30. Pull Request Strategy for Issue #1

Source:

1-project-initialization-and-django-setup

Target:

develop

Suggested title:

feat: complete Django project initialization

Suggested description:

## Summary

- Configured Django environment variables
- Configured PostgreSQL
- Configured Django REST Framework
- Added Django template and static file support
- Added application health endpoint
- Added initial dashboard
- Added basic automated tests
- Simplified dependency management to a single requirements.txt

## Validation

- `python manage.py check`
- `python manage.py test`
- `python manage.py migrate`

Closes #1

After merging:

git checkout develop
git pull origin develop
git branch -d 1-project-initialization-and-django-setup
git push origin --delete 1-project-initialization-and-django-setup

31. Issue Roadmap

Issue #1 — Project initialization and Django setup

Current issue.

Includes:

repository

Django

PostgreSQL

environment variables

DRF

health endpoint

dashboard

tests

Issue #2 — Dataset ingestion and data profiling

C-MAPSS ingestion

schema validation

data profiling

EDA

dataset documentation

Issue #3 — Leakage-safe preprocessing and feature engineering

split strategy

machine/group awareness

temporal considerations

rolling features

lag/delta features

scaling

Issue #4 — Classical ML failure-risk baseline

Logistic Regression

Random Forest

XGBoost

confusion matrix

recall/F1

PR-AUC

Issue #5 — RUL regression baseline

regression baselines

MAE

RMSE

R²

error analysis

Issue #6 — Custom PyTorch training pipeline

Dataset

DataLoader

MLP

training/validation loops

checkpointing

early stopping

weighted loss

Issue #7 — Temporal deep learning

sequence creation

1D CNN

LSTM

model comparison

Issue #8 — Explainability and error analysis

SHAP

sensor importance

false-negative analysis

operating-condition analysis

Issue #9 — ML experiment tracking

MLflow

parameters

metrics

artifacts

model versions

Issue #10 — Prediction API

Potential:

POST /api/predictions/

Input/output schema should be finalized after the ML pipeline exists.

Issue #11 — PostgreSQL machine/prediction models

Potential entities:

machines

sensor observations where appropriate

predictions

model metadata

Do not blindly store huge raw time-series data in relational tables without an architectural decision.

Issue #12 — Dashboard integration

Potential:

machine list

risk

RUL

prediction history

model information

explanations

Issue #13 — Dockerization

Containerize development environment once useful.

Issue #14 — Production hardening

Potential:

Gunicorn

logging

security settings

health/readiness checks

CI/CD

deployment

32. Definition of Done

For software issues:

[ ] Requirement understood
[ ] Design decided
[ ] Implementation complete
[ ] Tests added
[ ] Local verification complete
[ ] Documentation updated
[ ] Git diff reviewed
[ ] Commit created
[ ] Branch pushed
[ ] Pull Request created
[ ] PR reviewed
[ ] PR merged into develop
[ ] Issue closed

For ML issues:

[ ] Dataset assumptions documented
[ ] Leakage checked
[ ] Baseline established
[ ] Metrics selected
[ ] Experiment reproducible
[ ] Error analysis completed
[ ] Model artifact/version recorded

33. Current Project State

At the latest verified point:

GitHub repository          ✅
main                       ✅
develop                    ✅
Issue branch               ✅
Django project             ✅
Virtual environment        ✅
PostgreSQL 16              ✅
sentinel database          ✅
sentinel_user              ✅
.env configuration         ✅
python manage.py check     ✅
python manage.py migrate   ✅
single requirements.txt    ⏳ finalize/commit
health endpoint            ⏳
dashboard                  ⏳
automated test             ⏳
push issue branch          ⏳
PR → develop               ⏳

PostgreSQL + Django connectivity has already been verified successfully by migrations.

34. Immediate Next Steps

Finish Issue #1 in this order:

1. Finalize requirements.txt
2. Review config/settings.py
3. Configure DRF
4. Add health endpoint
5. Add dashboard
6. Add static CSS
7. Add health test
8. Run:
       python manage.py check
       python manage.py test
       python manage.py migrate
9. Review:
       git status
       git diff
10. Commit logical changes
11. Push issue branch
12. Create PR → develop
13. Merge
14. Close Issue #1
15. Start Issue #2

35. Reusable New-Project Checklist

Business

[ ] What business problem are we solving?
[ ] Who is the user?
[ ] What decision does the system improve?
[ ] What is the measurable outcome?
[ ] Functional requirements?
[ ] Non-functional requirements?

Scope

[ ] Define MVP
[ ] Define out-of-scope items
[ ] Convert future work into issues

Architecture

[ ] Define components
[ ] Define data flow
[ ] Choose database
[ ] Define APIs
[ ] Define ML components
[ ] Avoid unnecessary infrastructure

Repository

[ ] GitHub repository
[ ] README
[ ] .gitignore
[ ] .env.example
[ ] virtual environment
[ ] dependency file
[ ] project structure

Git

[ ] main
[ ] develop
[ ] Issues
[ ] issue branches
[ ] logical commits
[ ] push
[ ] PR → develop
[ ] merge
[ ] release → main

Backend

[ ] Environment configuration
[ ] Database
[ ] Framework configuration
[ ] Health endpoint
[ ] Tests
[ ] check
[ ] migrations
[ ] test suite

ML

[ ] Understand data
[ ] Validate data
[ ] Investigate leakage
[ ] Design split
[ ] Establish baseline
[ ] Feature engineering
[ ] Evaluate
[ ] Deep learning
[ ] Compare models
[ ] Error analysis
[ ] Explainability

Productionization

[ ] Inference API
[ ] Persistence
[ ] Model versioning
[ ] Experiment tracking
[ ] Logging/monitoring
[ ] Docker
[ ] CI/CD
[ ] Deployment

36. Golden Rules

Understand the business requirement before choosing technology.

Define the MVP before writing large amounts of code.

Explicitly document what is out of scope.

Start with a simple working baseline.

Never allow data leakage.

Never fabricate ML metrics.

Use metrics appropriate to the business problem.

Write tests for meaningful functionality.

Keep secrets out of Git.

Use issue/feature branches.

Merge feature branches into develop, not main.

Keep commits small and meaningful.

Do not add infrastructure just for appearance.

Do not create empty modules before they are needed.

Document important technical decisions.

Let experiments determine which model wins.

Treat ML as part of a software system, not as an isolated notebook.

Finish one issue properly before starting the next.

37. Reference Command Sheet

Python

python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Django

python manage.py check
python manage.py makemigrations
python manage.py migrate
python manage.py test
python manage.py runserver

PostgreSQL

Get-Service *postgres*
Get-ChildItem "C:\Program Files\PostgreSQL" -Recurse -Filter psql.exe -ErrorAction SilentlyContinue
& "C:\Program Files\PostgreSQLin\psql.exe" --version
& "C:\Program Files\PostgreSQLin\psql.exe" -U postgres

Inside psql:

\l
\du
\c sentinel

SELECT datname FROM pg_database;
SELECT rolname FROM pg_roles;
\q

VS Code

code .

Git

git status
git branch -vv
git log --oneline --decorate --graph --all
git fetch origin
git pull origin develop
git checkout develop
git checkout -b feature/<issue-number>-<description>
git add -A
git status
git diff --cached
git commit -m "type: description"
git push -u origin <branch-name>

38. Engineering Mindset

Build SENTINEL as a product, not as a collection of technologies:

Business Requirement
        ↓
MVP Scope
        ↓
Issue
        ↓
Design
        ↓
Implementation
        ↓
Test
        ↓
Experiment / Evidence
        ↓
Pull Request
        ↓
develop
        ↓
Release
        ↓
main

Avoid:

Technology
   ↓
Random code
   ↓
Huge repository
   ↓
No tests
   ↓
Fake metrics

The goal is to demonstrate engineering judgment, reproducibility, clean development workflow, and evidence-based ML decisions.