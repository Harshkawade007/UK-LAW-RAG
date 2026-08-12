"""
The test questions everything is scored against.

Every change to retrieval is measured against this same list, so "it got
better" is a number rather than an impression. eval/run_eval.py reads it.

Each question is grounded in the corpus: the fact it asks about really is in
the text at expected_parent_ids. That means a miss is a genuine gap in the
search, not an unanswerable question. The few questions with no expected
sections are there on purpose - the corpus genuinely cannot answer them, and
the right behaviour is to decline rather than improvise.

⚠️ These questions are tied to the exact corpus in laws/. The section ids below
are positions within a page, so re-fetching the pages would shift them and the
scores would quietly stop meaning anything. That is why laws/ is committed to
the repository instead of being downloaded fresh.

Each entry has these fields:

    question            a plainly-phrased question, in the words a student
                        would actually use
    category            which of the seven categories it belongs to (a list,
                        since a few genuinely span two)
    expected_sources    the page URLs the answer lives on. For reference only -
                        NOT used for scoring.
    expected_parent_ids the exact sections that contain the answer. THIS is
                        what gets scored.
    must_mention        short phrases a faithful answer should contain
    notes               why the question is here and what it is meant to catch

Why scoring uses sections rather than page URLs: a single page can have dozens
of sections that all share one URL, so scoring by URL gives full marks for
finding any part of the right page. Measured on this very set, page-level
scoring reported no change at all between two pipelines, while the actual
sections returned changed on 17 of 18 questions - including moving the correct
section from third place to first. The metric simply could not see it.

Every section id here was checked by reading the actual text, not guessed from
the page title. The per-question notes record what was checked.
"""

TESTSET = [
    # --- visa -------------------------------------------------------------
    {
        "question": "What can I not do on a Student visa?",
        "category": ["visa"],
        "expected_sources": ["https://www.gov.uk/student-visa"],
        "expected_parent_ids": ["visa/student-visa#6"],
        "must_mention": ["public funds", "self-employed"],
        "notes": "Direct fact lookup. #6 'What you can and cannot do' is the only section listing restrictions.",
    },
    {
        "question": "How many hours a week can I work on a Student visa during term time?",
        "category": ["visa"],
        "expected_sources": ["https://www.gov.uk/student-visa"],
        "expected_parent_ids": ["visa/student-visa#6"],
        "must_mention": [],
        "notes": (
            "KNOWN GAP IN THE CORPUS: #6 is the only section that discusses work "
            "hours at all, and it only says hours 'depend on what you're "
            "studying' - the '20 hours' figure never appears on this page. The "
            "right behaviour is to decline to state a number rather than borrow "
            "one from another visa route. Fixing this needs more pages fetched, "
            "not a better search."
        ),
    },
    {
        "question": "Are BRP (biometric residence permit) cards still valid?",
        "category": ["visa"],
        "expected_sources": ["https://www.gov.uk/biometric-residence-permits"],
        "expected_parent_ids": ["visa/biometric-residence-permits#0"],
        "must_mention": ["expired", "eVisa"],
        "notes": "Tests a specific, date-sensitive fact. The 'all BRPs have now expired, replaced by eVisas' sentence is in the page intro (#0), not a subsection.",
    },
    {
        "question": "I'm on a Student visa - can I switch to a Skilled Worker visa after I graduate?",
        "category": ["visa", "employment"],
        "expected_sources": [
            "https://www.gov.uk/student-visa",
            "https://www.gov.uk/skilled-worker-visa",
        ],
        "expected_parent_ids": [
            "visa/skilled-worker-visa#72",
            "visa/skilled-worker-visa#73",
        ],
        "must_mention": [],
        "notes": (
            "Needs two hops: the switching rules live on the SKILLED WORKER page "
            "('Switch to this visa' #72 and 'Eligibility' #73, which explicitly "
            "covers switching from a Student visa), not on the student-visa page "
            "the question sounds like it is about."
        ),
    },

    # --- tax_ni -------------------------------------------------------------
    {
        "question": "Who is eligible to apply for a National Insurance number?",
        "category": ["tax_ni"],
        "expected_sources": ["https://www.gov.uk/apply-national-insurance-number"],
        "expected_parent_ids": ["tax_ni/apply-national-insurance-number#1"],
        "must_mention": ["right to work"],
        "notes": "Direct fact lookup, 'Who can apply' section.",
    },
    {
        "question": "Can I start working before I get my National Insurance number?",
        "category": ["tax_ni"],
        "expected_sources": ["https://www.gov.uk/apply-national-insurance-number"],
        "expected_parent_ids": ["tax_ni/apply-national-insurance-number#1"],
        "must_mention": ["right to work"],
        "notes": (
            "Same PARENT section as above, not just the same page - verified the "
            "exact sentence ('You can start work before you receive your National "
            "Insurance number if you can prove you have the right to work') sits "
            "in #1 alongside the eligibility list. Tests it isn't a fluke match."
        ),
    },
    {
        "question": "What is the Personal Allowance for Income Tax?",
        "category": ["tax_ni"],
        "expected_sources": ["https://www.gov.uk/income-tax-rates"],
        "expected_parent_ids": ["tax_ni/income-tax-rates#2"],
        "must_mention": [],
        "notes": "Numeric fact; 'Your tax-free Personal Allowance' section, not the rates-and-bands table in #3.",
    },

    # --- housing -------------------------------------------------------------
    {
        "question": "Do full-time students have to pay Council Tax?",
        "category": ["housing"],
        "expected_sources": ["https://www.gov.uk/council-tax"],
        "expected_parent_ids": ["housing/council-tax#6"],
        "must_mention": ["full-time student"],
        "notes": "Confirmed working well in manual testing. 'Discounts for full-time students' section specifically.",
    },
    {
        "question": "Does my landlord have to protect my deposit?",
        "category": ["housing"],
        "expected_sources": [
            "https://www.gov.uk/tenancy-deposit-protection",
            "https://www.gov.uk/private-renting",
        ],
        "expected_parent_ids": [
            "housing/tenancy-deposit-protection#1",
            "housing/private-renting#32",
        ],
        "must_mention": ["deposit protection scheme"],
        "notes": (
            "Direct fact lookup. TWO sections genuinely answer this and both "
            "count: tenancy-deposit-protection#1 ('Your landlord must put your "
            "deposit in a government-approved tenancy deposit scheme') and "
            "private-renting#32 ('In England, your landlord must keep your "
            "deposit safe using a government-approved tenancy deposit "
            "protection scheme'). private-renting#32 was added 2026-08-05 "
            "after it was scored a miss despite materially answering the "
            "question - a grounding bug, not a retrieval failure.\n"
            "This question is also the corpus's best trap for score-based "
            "relevance: `tenancy-deposit-protection#4` 'Holding deposits' says "
            "the landlord does NOT have to protect a holding deposit, and the "
            "cross-encoder scores it 9.34 - near the highest in the testset - "
            "while it answers the opposite question. See agent/grade.py."
        ),
    },
    {
        "question": "What types of tenancy exist for private renting in the UK?",
        "category": ["housing"],
        "expected_sources": ["https://www.gov.uk/private-renting"],
        "expected_parent_ids": ["housing/private-renting#1"],
        "must_mention": ["tenancy"],
        "notes": "Overview-style question - 'Tenancy types' section on a 47-section page, exactly the kind of within-page distinction page-level scoring couldn't see.",
    },

    # --- employment -------------------------------------------------------------
    {
        "question": "What is the National Minimum Wage and does age affect it?",
        "category": ["employment"],
        "expected_sources": ["https://www.gov.uk/national-minimum-wage-rates"],
        "expected_parent_ids": [
            "employment/national-minimum-wage-rates#0",
            "employment/national-minimum-wage-rates#1",
        ],
        "must_mention": ["age"],
        "notes": "Split across two parents: #0 (intro) states the rate depends on age, #1 has the actual age-banded rate table. Either answers it.",
    },
    {
        "question": "How many weeks of paid holiday am I entitled to as a worker?",
        "category": ["employment"],
        "expected_sources": ["https://www.gov.uk/holiday-entitlement-rights"],
        "expected_parent_ids": ["employment/holiday-entitlement-rights#1"],
        "must_mention": ["5.6 weeks"],
        "notes": "Exact-number fact - good BM25/hybrid-search comparison case. 'Entitlement' section, not the 23-section pay-calculation detail further down the page.",
    },

    # --- education -------------------------------------------------------------
    {
        "question": "What can I use student finance to help pay for?",
        "category": ["education"],
        "expected_sources": ["https://www.gov.uk/student-finance"],
        "expected_parent_ids": ["education/student-finance#1"],
        "must_mention": [],
        "notes": "Overview-style question - the page Overview lists Tuition Fee Loan / Maintenance Loan / what each covers.",
    },
    {
        "question": "Can I get student finance if I've studied a degree before?",
        "category": ["education"],
        "expected_sources": ["https://www.gov.uk/student-finance"],
        "expected_parent_ids": ["education/student-finance#7"],
        "must_mention": [],
        "notes": (
            "Section #7 ('If you've studied before') is one that chunk.py splits "
            "into three overlapping chunks, so this checks the split did not "
            "break retrieval of the fact. It also shows section-level scoring "
            "working: reranking moves this from third place to first."
        ),
    },

    # --- banking -------------------------------------------------------------
    {
        "question": "What is Attendance Allowance and who is it for?",
        "category": ["banking"],
        "expected_sources": ["https://www.gov.uk/attendance-allowance"],
        "expected_parent_ids": [
            "banking/attendance-allowance#1",
            "banking/attendance-allowance#6",
        ],
        "must_mention": ["disability"],
        "notes": "Two-part question, two-part answer: 'Overview' (#1, what it is) + 'Eligibility' (#6, who it's for). Also contains a rate table, good structure test.",
    },

    # --- nhs -------------------------------------------------------------
    {
        "question": "What does the NHS entitlements migrant health guide cover?",
        "category": ["nhs"],
        "expected_sources": [
            "https://www.gov.uk/guidance/nhs-entitlements-migrant-health-guide"
        ],
        "expected_parent_ids": [
            "nhs/guidance__nhs-entitlements-migrant-health-guide#1"
        ],
        "must_mention": [],
        "notes": (
            "Direct fact lookup from the one gov.uk (not nhs.uk) NHS source. "
            "'What this guidance is for' section is an exact match for the "
            "question.\n"
            "REVIEWED 2026-08-05 and deliberately LEFT STRICT. Retrieval "
            "returns #5 'Free services' / #3 'Brexit changes' / #12 "
            "'Vulnerable migrants' - those are contents OF the guide, but the "
            "question asks what the guide covers, and #1 is the section that "
            "actually describes its purpose and scope. Unlike the deposit "
            "question, the retrieved sections do not answer what was asked, so "
            "this stays a genuine miss rather than a grounding bug."
        ),
    },
    {
        "question": "How do I register with a GP surgery in England?",
        "category": ["nhs"],
        "expected_sources": [
            "https://www.nhs.uk/nhs-services/gps/how-to-register-with-a-gp-surgery/"
        ],
        "expected_parent_ids": [
            "nhs/nhs-services__gps__how-to-register-with-a-gp-surgery#0",
            "nhs/nhs-services__gps__how-to-register-with-a-gp-surgery#1",
        ],
        "must_mention": ["free"],
        "notes": (
            "From the nhs.uk scrape (fetch_nhs.py), not gov.uk - tests both "
            "source types. 'for free' is in the page intro (#0), not the "
            "'Register or change GP surgery online' how-to steps (#1) - "
            "either section satisfies the question."
        ),
    },

    # =====================================================================
    # Batch 2 - deliberately harder than the questions above.
    #
    # The first batch was written using wording close to gov.uk's own, so most
    # of those questions already came back in first place. That left no room
    # to tell a genuine improvement from random variation - one pipeline was
    # measured swinging more between repeat runs than the differences being
    # compared.
    #
    # So these are written the way a student would actually type them ("my
    # student visa runs out next month...", "my new boss wants proof"), rather
    # than the way gov.uk writes its headings. Several are traps on purpose:
    # negations, near-identical wording with opposite answers, and questions
    # that look like they are about one thing but are answered by another.
    #
    # Every section id below was checked by reading the actual text.
    # =====================================================================

    # --- visa: Graduate route -------------------------------------------------
    {
        "question": "After I finish my course can I stay in the UK and look for a job?",
        "category": ["visa"],
        "expected_sources": ["https://www.gov.uk/graduate-visa"],
        "expected_parent_ids": ["visa/graduate-visa#1", "visa/graduate-visa#7"],
        "must_mention": [],
        "notes": (
            "Colloquial phrasing of the Graduate visa. Never says 'Graduate "
            "visa', so pure keyword matching cannot find it - the system has "
            "to connect 'stay after my course' to the route's name. #1 states "
            "the 18-month permission, #7 says you can 'look for work'."
        ),
    },
    {
        "question": "My student visa runs out next month - can I still apply for the graduate route?",
        "category": ["visa"],
        "expected_sources": ["https://www.gov.uk/graduate-visa"],
        "expected_parent_ids": ["visa/graduate-visa#5"],
        "must_mention": [],
        "notes": (
            "TRAP - timing constraint. #5: 'You must apply before your Student "
            "visa or Tier 4 (General) student visa expires.' The correct answer "
            "hinges on a deadline, and a system that retrieves the general "
            "'How to apply' section instead will miss it entirely."
        ),
    },
    {
        "question": "How long does a graduate visa last if I have a PhD?",
        "category": ["visa"],
        "expected_sources": ["https://www.gov.uk/graduate-visa"],
        "expected_parent_ids": ["visa/graduate-visa#3"],
        "must_mention": [],
        "notes": (
            "Conditional numeric fact: #3 gives 2 years / 18 months by "
            "application date, then 3 years specifically for a doctoral "
            "qualification. Tests retrieval of the qualifying clause, not just "
            "the headline number."
        ),
    },

    # --- nhs: immigration health surcharge ------------------------------------
    {
        "question": "Do I have to pay something for the NHS when I apply for my visa?",
        "category": ["nhs", "visa"],
        "expected_sources": ["https://www.gov.uk/healthcare-immigration-application"],
        "expected_parent_ids": [
            "nhs/healthcare-immigration-application#1",
            "nhs/healthcare-immigration-application#3",
        ],
        "must_mention": [],
        "notes": "Student never says 'immigration health surcharge' or 'IHS' - the vocabulary gap is the whole test. #1 overview, #3 who needs to pay.",
    },
    {
        "question": "How much is the health surcharge per year for a student?",
        "category": ["nhs", "visa"],
        "expected_sources": ["https://www.gov.uk/healthcare-immigration-application"],
        "expected_parent_ids": ["nhs/healthcare-immigration-application#6"],
        "must_mention": [],
        "notes": "Exact-number fact (GBP 776 per year for students) buried in a page with several cost sections (#5 #6 #7 #8) - good BM25-vs-dense discrimination case.",
    },
    {
        "question": "When can I actually start using the NHS for free after I arrive?",
        "category": ["nhs", "visa"],
        "expected_sources": ["https://www.gov.uk/healthcare-immigration-application"],
        "expected_parent_ids": ["nhs/healthcare-immigration-application#2"],
        "must_mention": [],
        "notes": "#2: free 'from the date your visa starts', conditional on having paid the IHS. Tests a timing answer on a page dominated by cost sections.",
    },

    # --- employment -----------------------------------------------------------
    {
        "question": "My new boss wants proof I'm allowed to work here - what do I show them?",
        "category": ["employment"],
        "expected_sources": ["https://www.gov.uk/prove-right-to-work"],
        "expected_parent_ids": ["employment/prove-right-to-work#3"],
        "must_mention": [],
        "notes": "Colloquial. #3 covers non-British/Irish citizens: a share code or eligible immigration documents. The relevant branch for an international student.",
    },
    {
        "question": "Can my employer take money out of my wages?",
        "category": ["employment"],
        "expected_sources": ["https://www.gov.uk/understanding-your-pay"],
        "expected_parent_ids": ["employment/understanding-your-pay#14"],
        "must_mention": [],
        "notes": "#14 'Deductions from your pay' - lists when deductions are allowed. Plain-language phrasing of a section titled with jargon ('deductions').",
    },
    {
        "question": "I work part time - should I get the same hourly rate as full time staff?",
        "category": ["employment"],
        "expected_sources": ["https://www.gov.uk/understanding-your-pay"],
        "expected_parent_ids": ["employment/understanding-your-pay#2"],
        "must_mention": [],
        "notes": (
            "Very short target section (one sentence: part-time workers must "
            "get at least the same hourly rate). Short sections are easy to "
            "lose against longer, wordier ones that share vocabulary - a "
            "deliberate chunk-length stress test."
        ),
    },

    # --- tax_ni: student loan repayment ---------------------------------------
    {
        "question": "When do I actually start paying back my student loan?",
        "category": ["tax_ni"],
        "expected_sources": ["https://www.gov.uk/repaying-your-student-loan"],
        "expected_parent_ids": ["tax_ni/repaying-your-student-loan#12"],
        "must_mention": [],
        "notes": (
            "#12 'When you start repaying' - income over your plan's "
            "threshold. Hard because the same page has five near-identical "
            "plan-specific sections (#13-#17) all containing threshold "
            "numbers; the general answer must beat them."
        ),
    },
    {
        "question": "Will I get charged extra if I pay my student loan off early?",
        "category": ["tax_ni"],
        "expected_sources": ["https://www.gov.uk/repaying-your-student-loan"],
        "expected_parent_ids": ["tax_ni/repaying-your-student-loan#18"],
        "must_mention": [],
        "notes": (
            "NEGATION test. #18 is a single sentence: 'There's no penalty for "
            "paying some or all of your loan off early.' The answer is 'no', "
            "and a system that retrieves the interest sections (#23, #24) "
            "instead will produce a confidently wrong answer."
        ),
    },
    {
        "question": "I'm going home for 6 months - do I need to tell anyone about my student loan?",
        "category": ["tax_ni"],
        "expected_sources": ["https://www.gov.uk/repaying-your-student-loan"],
        "expected_parent_ids": ["tax_ni/repaying-your-student-loan#3"],
        "must_mention": [],
        "notes": "#3: must tell the Student Loans Company if leaving the UK for more than 3 months. Tests matching '6 months' against a 'more than 3 months' rule.",
    },

    # --- housing --------------------------------------------------------------
    {
        "question": "If I rent out my spare room, do I have to protect my lodger's deposit?",
        "category": ["housing"],
        "expected_sources": ["https://www.gov.uk/rent-room-in-your-home"],
        "expected_parent_ids": ["housing/rent-room-in-your-home#16"],
        "must_mention": [],
        "notes": (
            "THE KEY TRAP IN THIS SET. Near-identical vocabulary to 'Does my "
            "landlord have to protect my deposit?' but the OPPOSITE answer: "
            "#16 says 'Resident landlords are not legally required to protect "
            "tenants' deposit'. Any system that matches on deposit-protection "
            "wording alone will retrieve the tenancy-deposit-protection page "
            "and answer exactly backwards. This is the same failure shape as "
            "the 'Holding deposits' case that motivated agent/grade.py."
        ),
    },
    {
        "question": "I've got a lodger living with me - who pays the council tax?",
        "category": ["housing"],
        "expected_sources": ["https://www.gov.uk/rent-room-in-your-home"],
        "expected_parent_ids": ["housing/rent-room-in-your-home#12"],
        "must_mention": [],
        "notes": (
            "Cross-cutting: a Council Tax question whose answer is NOT on the "
            "council-tax page. Also tests the single-person-discount caveat. "
            "Good check that routing does not over-narrow to the obvious page."
        ),
    },
    {
        "question": "My landlord wants me to move out - what do they actually have to do first?",
        "category": ["housing"],
        "expected_sources": ["https://www.gov.uk/private-renting-evictions"],
        "expected_parent_ids": ["housing/private-renting-evictions#1"],
        "must_mention": [],
        "notes": "#1 'Rules your landlord must follow' - strict procedures, illegal eviction if not followed. Colloquial phrasing of an eviction-procedure question.",
    },

    # --- visa: TB test + border ------------------------------------------------
    {
        "question": "Do I need a TB test before I come to the UK to study?",
        "category": ["visa"],
        "expected_sources": ["https://www.gov.uk/tb-test-visa"],
        "expected_parent_ids": ["visa/tb-test-visa#1"],
        "must_mention": [],
        "notes": "#1 gives the conditions (6 months or more + lived in a listed country). Page is mostly A-Z country lists (#8-#15), which are strong lexical distractors.",
    },
    {
        "question": "Who is exempt from the TB test?",
        "category": ["visa"],
        "expected_sources": ["https://www.gov.uk/tb-test-visa"],
        "expected_parent_ids": ["visa/tb-test-visa#3"],
        "must_mention": [],
        "notes": (
            "Uses 'exempt', a word that does NOT appear in the target section "
            "(#3 is headed 'Who does not need to be tested'). Pure synonym "
            "test - BM25 cannot bridge it, so this isolates the semantic half "
            "of hybrid search."
        ),
    },
    {
        "question": "What happens at UK border control when I arrive?",
        "category": ["visa"],
        "expected_sources": ["https://www.gov.uk/uk-border-control"],
        "expected_parent_ids": ["visa/uk-border-control#10", "visa/uk-border-control#14"],
        "must_mention": [],
        "notes": "#10 general border control steps; #14 the non-EU route relevant to most international students. Either answers it.",
    },

    # --- education -------------------------------------------------------------
    {
        "question": "Can I get help paying for childcare while I'm studying?",
        "category": ["education"],
        "expected_sources": ["https://www.gov.uk/childcare-grant"],
        "expected_parent_ids": ["education/childcare-grant#1", "education/childcare-grant#6"],
        "must_mention": [],
        "notes": "#1 overview / #6 eligibility. Adds coverage to `education`, the thinnest category in the corpus (214 children).",
    },
    {
        "question": "How much childcare grant can I get for two kids?",
        "category": ["education"],
        "expected_sources": ["https://www.gov.uk/childcare-grant"],
        "expected_parent_ids": ["education/childcare-grant#3", "education/childcare-grant#4"],
        "must_mention": [],
        "notes": (
            "AMBIGUITY test: the page holds two near-identical rate sections, "
            "#3 (2026-27) and #4 (2025-26), differing only by academic year "
            "and amounts. The question does not specify a year, so both count "
            "- but a good system should surface a current-year rate rather "
            "than an arbitrary one."
        ),
    },

    # --- questions the corpus cannot answer ----------------------------------
    # These have no expected sections, so they are excluded from the scores.
    # They are here to check the system declines rather than improvising an
    # answer out of whatever text happens to be closest.
    # -------------------------------------------------------------------------
    {
        "question": "Can I bring my pet dog with me when I move to the UK to study?",
        "category": ["visa"],
        "expected_sources": [],
        "expected_parent_ids": [],
        "must_mention": [],
        "notes": (
            "Deliberately not covered by any page in the corpus. The right "
            "outcome is an honest 'I don't have anything on this', not a "
            "confident guess assembled from nearby visa sections."
        ),
    },
    {
        "question": "Which UK university is best for computer science?",
        "category": ["education"],
        "expected_sources": [],
        "expected_parent_ids": [],
        "must_mention": [],
        "notes": (
            "Checked: not a single chunk in the corpus mentions university "
            "rankings or league tables. Unlike the pet-dog question this is a "
            "very plausible student question - it is just about something "
            "gov.uk does not publish an opinion on, so the only honest answer "
            "is to say so."
        ),
    },
]


if __name__ == "__main__":
    from collections import Counter
    cats = Counter(c for q in TESTSET for c in q["category"])
    print(f"{len(TESTSET)} questions")
    for cat, n in cats.most_common():
        print(f"  {cat:12} {n}")
