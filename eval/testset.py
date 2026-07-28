"""
eval/testset.py

The golden test questions: your baseline ruler. Every future retrieval change
(hybrid search, reranking, routing) gets measured against this same set, so
"we improved it" is a number, not a feeling.

Each question is grounded in a real page in the corpus - the fact it asks
about is actually present at expected_sources - so a wrong or missing answer
is diagnosable as a genuine pipeline gap, not an unanswerable question.

Fields:
    question         - a real, plainly-phrased student question
    category         - which of the 7 corpus categories it belongs to
                       (a list, since a few genuinely span two)
    expected_sources - source_url(s) whose content should answer it; used to
                       score RETRIEVAL (did the right page make top-k?)
    must_mention     - short substrings the final ANSWER should contain if
                       generation is faithful to that source (case-insensitive)
    notes            - why this question is here / what it's meant to catch
"""

TESTSET = [
    # --- visa -------------------------------------------------------------
    {
        "question": "What can I not do on a Student visa?",
        "category": ["visa"],
        "expected_sources": ["https://www.gov.uk/student-visa"],
        "must_mention": ["public funds", "self-employed"],
        "notes": "Direct fact lookup, single well-covered section.",
    },
    {
        "question": "How many hours a week can I work on a Student visa during term time?",
        "category": ["visa"],
        "expected_sources": ["https://www.gov.uk/student-visa"],
        "must_mention": [],
        "notes": (
            "KNOWN COVERAGE GAP: the '20 hours' figure is not present on the "
            "fetched student-visa page. Expect the model to correctly decline "
            "to state a number rather than borrow the Skilled Worker visa's 20h "
            "or invent one. Candidate for --discover / a targeted re-fetch."
        ),
    },
    {
        "question": "Are BRP (biometric residence permit) cards still valid?",
        "category": ["visa"],
        "expected_sources": ["https://www.gov.uk/biometric-residence-permits"],
        "must_mention": ["expired", "eVisa"],
        "notes": "Tests a specific, date-sensitive fact.",
    },
    {
        "question": "I'm on a Student visa - can I switch to a Skilled Worker visa after I graduate?",
        "category": ["visa", "employment"],
        "expected_sources": [
            "https://www.gov.uk/student-visa",
            "https://www.gov.uk/skilled-worker-visa",
        ],
        "must_mention": [],
        "notes": "Multi-hop: needs both visa pages. Good candidate for Week 3 agent test.",
    },

    # --- tax_ni -------------------------------------------------------------
    {
        "question": "Who is eligible to apply for a National Insurance number?",
        "category": ["tax_ni"],
        "expected_sources": ["https://www.gov.uk/apply-national-insurance-number"],
        "must_mention": ["right to work"],
        "notes": "Direct fact lookup.",
    },
    {
        "question": "Can I start working before I get my National Insurance number?",
        "category": ["tax_ni"],
        "expected_sources": ["https://www.gov.uk/apply-national-insurance-number"],
        "must_mention": ["right to work"],
        "notes": "Same page as above, different angle - tests it isn't a fluke match.",
    },
    {
        "question": "What is the Personal Allowance for Income Tax?",
        "category": ["tax_ni"],
        "expected_sources": ["https://www.gov.uk/income-tax-rates"],
        "must_mention": [],
        "notes": "Numeric fact; a table-derived answer.",
    },

    # --- housing -------------------------------------------------------------
    {
        "question": "Do full-time students have to pay Council Tax?",
        "category": ["housing"],
        "expected_sources": ["https://www.gov.uk/council-tax"],
        "must_mention": ["full-time student"],
        "notes": "Confirmed working well in manual testing.",
    },
    {
        "question": "Does my landlord have to protect my deposit?",
        "category": ["housing"],
        "expected_sources": ["https://www.gov.uk/tenancy-deposit-protection"],
        "must_mention": ["deposit protection scheme"],
        "notes": "Direct fact lookup.",
    },
    {
        "question": "What types of tenancy exist for private renting in the UK?",
        "category": ["housing"],
        "expected_sources": ["https://www.gov.uk/private-renting"],
        "must_mention": ["tenancy"],
        "notes": "Overview-style question - broader parent section.",
    },

    # --- employment -------------------------------------------------------------
    {
        "question": "What is the National Minimum Wage and does age affect it?",
        "category": ["employment"],
        "expected_sources": ["https://www.gov.uk/national-minimum-wage-rates"],
        "must_mention": ["age"],
        "notes": "Numeric/table fact.",
    },
    {
        "question": "How many weeks of paid holiday am I entitled to as a worker?",
        "category": ["employment"],
        "expected_sources": ["https://www.gov.uk/holiday-entitlement-rights"],
        "must_mention": ["5.6 weeks"],
        "notes": "Exact-number fact - good BM25/hybrid-search comparison case.",
    },

    # --- education -------------------------------------------------------------
    {
        "question": "What can I use student finance to help pay for?",
        "category": ["education"],
        "expected_sources": ["https://www.gov.uk/student-finance"],
        "must_mention": [],
        "notes": "Overview-style question.",
    },
    {
        "question": "Can I get student finance if I've studied a degree before?",
        "category": ["education"],
        "expected_sources": ["https://www.gov.uk/student-finance"],
        "must_mention": [],
        "notes": (
            "This is the exact section chunk.py splits into 3 overlapping "
            "children (parent #7) - good regression check that the split "
            "didn't break retrieval of this fact."
        ),
    },

    # --- banking -------------------------------------------------------------
    {
        "question": "What is Attendance Allowance and who is it for?",
        "category": ["banking"],
        "expected_sources": ["https://www.gov.uk/attendance-allowance"],
        "must_mention": ["disability"],
        "notes": "Direct fact lookup; also contains a rate table, good structure test.",
    },

    # --- nhs -------------------------------------------------------------
    {
        "question": "What does the NHS entitlements migrant health guide cover?",
        "category": ["nhs"],
        "expected_sources": [
            "https://www.gov.uk/guidance/nhs-entitlements-migrant-health-guide"
        ],
        "must_mention": [],
        "notes": "Direct fact lookup from the one gov.uk (not nhs.uk) NHS source.",
    },
    {
        "question": "How do I register with a GP surgery in England?",
        "category": ["nhs"],
        "expected_sources": [
            "https://www.nhs.uk/nhs-services/gps/how-to-register-with-a-gp-surgery/"
        ],
        "must_mention": ["free"],
        "notes": "From the nhs.uk scrape (fetch_nhs.py), not gov.uk - tests both source types.",
    },

    # --- out-of-corpus: tests honest refusal, not retrieval -------------------
    {
        "question": "Can I bring my pet dog with me when I move to the UK to study?",
        "category": ["visa"],
        "expected_sources": [],
        "must_mention": [],
        "notes": (
            "Deliberately NOT covered by any seed page. The correct behaviour "
            "is an honest 'not confident' answer, not a confident guess. "
            "Tests the faithfulness rule, not retrieval."
        ),
    },
]


if __name__ == "__main__":
    from collections import Counter
    cats = Counter(c for q in TESTSET for c in q["category"])
    print(f"{len(TESTSET)} questions")
    for cat, n in cats.most_common():
        print(f"  {cat:12} {n}")
