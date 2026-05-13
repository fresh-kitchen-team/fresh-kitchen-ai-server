# Git Commit / PR Convention

## 1. Purpose

This document defines commit message, branch naming, and pull request conventions.

The goals are:

- consistent commit history
- readable logs for developers
- predictable structure for collaboration
- compatibility with AI-based tooling and automation

---

## 2. Branch Strategy

feat/* → main

### Branch Roles

- `feat/*`
  - feature or task-level branch
  - created per unit of work

- `main`
  - production-ready branch
  - deployment only

---

## 3. Branch Naming Convention

feat/domain-feature

### Rules

- lowercase only
- words separated by `-`
- short but meaningful
- reflect main purpose

### Examples

feat/food-classifier
feat/receipt-ocr
feat/fridge-detection
feat/gemini-fallback

---

## 4. Commit Message Convention

### 4.1 Format

Type(Scope) : Description

### Structure

- `Type`
  - category of change
  - uppercase first letter

- `Scope`
  - domain/module name
  - uppercase first letter

- `Description`
  - short summary in Korean
  - imperative form
  - no period at the end

---

### 4.2 Writing Rules

#### Language

- Type : English only
- Description : Korean only

#### Verb Style

- Use **imperative mood**

올바른 예시:

Add(Classifier) : 음식 분류 예측 함수 추가
Fix(Ocr) : Gemini 응답 파싱 오류 수정

잘못된 예시:

added classifier
fixing bug
create model

---

### 4.3 Allowed Types

| Type       | Description                         |
|------------|-------------------------------------|
| Feat       | new feature                         |
| Fix        | bug fix                             |
| Add        | add code/resource                   |
| Remove     | delete code/resource                |
| Refactor   | internal improvement                |
| Docs       | documentation                       |
| Chore      | maintenance                         |
| Test       | test code                           |
| Style      | formatting only                     |
| Implement  | realization of planned logic        |

---

### 4.4 Type Usage Guide

#### Feat

Feat(Classifier) : Gemini Vision 폴백 기능 추가

#### Fix

Fix(Ocr) : Document AI 응답 파싱 오류 수정

#### Add

Add(Training) : 클래스별 정확도 분석 로직 추가

#### Remove

Remove(Classifier) : 미사용 전처리 함수 삭제

#### Refactor

Refactor(Server) : 모델 로딩 방식 lifespan으로 전환

#### Docs

Docs(Readme) : API 엔드포인트 설명 추가

#### Chore

Chore(Model) : 모델 경로 ver3으로 업데이트

#### Test

Test(Classifier) : 클래스별 정확도 평가 스크립트 추가

#### Style

Style(Server) : 엔드포인트 주석 정리

#### Implement

Implement(Detection) : YOLOv8n 물체 감지 로직 적용

---

### 4.5 Scope Rules

Scope represents the main affected domain.

#### Examples

Classifier
Ocr
Detection
Training
Model
Server
Api
Data
Readme
Chore

#### Guidelines

- use one primary scope
- use domain-level naming
- avoid overly long scope names

---

## 5. Pull Request Convention

### Format

Type: Description

### Rules

- use one allowed `Type`
- do not include scope
- English only
- imperative mood
- capitalize the first letter of the type and description
- no period at the end

### Examples

Feat: Add Gemini Vision fallback for food classifier
Fix: Resolve Document AI response parsing error
Refactor: Simplify model loading logic
Docs: Update API endpoint description

---

## 6. Commit vs PR Roles

### Commit

- smallest logical change unit

Add(Classifier) : 음식 분류 예측 함수 추가
Fix(Ocr) : 영수증 한글 필터링 오류 수정

### PR

- summarizes entire feature branch
- uses no scope

Feat: Add Gemini Vision fallback for food classifier

---

## 7. Best Practices

### One Commit = One Intention

올바른 예시:

Add(Classifier) : Gemini 폴백 함수 추가
Test(Classifier) : 클래스별 정확도 평가 추가

잘못된 예시:

Add(Classifier) : Gemini 폴백 추가 및 테스트 수정 및 문서 업데이트

---

### Do Not Mix Concerns

Avoid mixing:

- feature logic
- refactoring
- tests
- documentation
- formatting

---

### Be Specific

올바른 예시:

Fix(Server) : 잘못된 형식의 Bearer 토큰 거부

잘못된 예시:

Fix(Server) : 오류 수정

---

## 8. Rules for AI and Automation

To ensure machine readability:

- always use exact format:

Type(Scope) : Description

- consistent capitalization
- no punctuation variation
- no mixed intentions
- PR format must follow:

Type: Description

### Benefits

- automated commit parsing
- changelog generation
- AI-assisted review
- consistent repository structure

---

## 9. Summary

### Commit

Type(Scope) : Description

### PR

Type: Description

### Branch

feat/domain-feature
