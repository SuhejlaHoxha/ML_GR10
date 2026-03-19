# Machine Learning 

> **Course:** Machine Learning  
> **Professor:** Prof. Dr. Lule AHMEDI  
> **Assistant:** Dr. Sc. Mërgim H. HOTI  
> **University:** University of Prishtina "Hasan Prishtina"  
> **Faculty:** Faculty of Electrical and Computer Engineering (FIEK)  
> **Study Level:** Master — Semester II | Academic Year: 2025/26

---

##  Contributors

| Name 
|------|
| `Dredhza Braina` |
| `Suhejla Hoxha` | 
| `Ylljete Kicaj` | 

---

## Project Overview

This project builds a complete **Phase I Data Preparation pipeline** for a GitHub activity-log dataset. The goal is to clean, transform, and prepare the data for training a binary classifier that predicts whether an actor is a **bot or a human** (`actor_is_bot`).

The entire pipeline is written in modular Python (no Jupyter notebooks) as a `src/` package.

---

##  Dataset Details

| Attribute | Value |
|-----------|-------|
| **Name** | GitHub Activity Log |
| **Original format** | JSON (flattened to CSV) |
| **Rows** | 10,000 |
| **Columns** | 590 |
| **Size** | ~45 MB in memory |
| **Source** | GitHub Audit Log API |
| **Target variable** | `actor_is_bot` (0 = human, 1 = bot) |
| **Time period** | Jan 10, 2026 — Mar 11, 2026 (59 days) |
| **Unique event types** | 481 |

**Key columns:** `@timestamp`, `_document_id`, `action`, `actor`, `actor_is_bot`, `operation_type`, `org`, `repo`, `visibility`, `actor_ip`

**Notable characteristics:**
- Only **4.36%** of all cells are filled — extremely sparse (flattened JSON)
- Class imbalance: **84.95% human** vs **15.05% bot**
- 572 columns have 100% missing values (event-specific fields)