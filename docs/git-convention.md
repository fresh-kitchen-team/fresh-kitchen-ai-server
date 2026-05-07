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

feat/user-auth
feat/bean-validation
feat/ingredient-management
feat/receipt-ocr-processing

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

Add(User) : 로그인 API 생성
Fix(Token) : 폐기된 토큰 액세스 방지

잘못된 예시:

added login api
fixing bug
create login

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

Feat(Auth) : 소셜 로그인 기능 추가

#### Fix

Fix(Token) : 만료된 토큰 처리 오류 수정

#### Add

Add(User) : 회원가입 DTO 추가

#### Remove

Remove(Auth) : 레거시 토큰 유틸 삭제

#### Refactor

Refactor(Bean) : 유효성 검사 로직 분리

#### Docs

Docs(Readme) : 설치 방법 문서 추가

#### Chore

Chore(Gradle) : 의존성 버전 업데이트

#### Test

Test(User) : 회원가입 유효성 테스트 추가

#### Style

Style(Api) : 컨트롤러 코드 포맷 정리

#### Implement

Implement(Receipt) : OCR 파싱 로직 적용

---

### 4.5 Scope Rules

Scope represents the main affected domain.

#### Examples

User
Auth
Bean
Receipt
Ingredient
Fridge
Api
Security
Readme
Gradle

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

Feat: Implement user signup flow
Fix: Resolve token validation bug
Refactor: Simplify bean validation logic
Docs: Update branch strategy

---

## 6. Commit vs PR Roles

### Commit

- smallest logical change unit

Add(User) : 회원가입 DTO 생성
Fix(User) : 중복 이메일 처리 오류 수정

### PR

- summarizes entire feature branch
- uses no scope

Feat: Implement user signup flow

---

## 7. Best Practices

### One Commit = One Intention

올바른 예시:

Add(Auth) : JWT provider 생성
Test(Auth) : JWT 테스트 추가

잘못된 예시:

Add(Auth) : JWT provider 생성 및 테스트 수정 및 문서 업데이트

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

Fix(Auth) : 잘못된 형식의 bearer 토큰 거부

잘못된 예시:

Fix(Auth) : 오류 수정

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
