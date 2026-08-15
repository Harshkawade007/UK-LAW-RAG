"""
The test questions everything is scored against.

Every change to retrieval is measured against this same list, so "it got
better" is a number rather than an impression. eval/run_eval.py reads it.

Each question is grounded in the corpus: the fact it asks about really is in
the text at expected_sections. That means a miss is a genuine gap in the
search, not an unanswerable question. The few questions with no expected
sections are there on purpose - the corpus genuinely cannot answer them, and
the right behaviour is to decline rather than improvise.

Each entry has these fields:

    question            a plainly-phrased question, in the words a student
                        would actually use
    category            which of the seven categories it belongs to (a list,
                        since a few genuinely span two)
    expected_sections    the sections that contain the answer, identified by
                        {"url": source_url, "heading": section heading or
                        None}. THIS is what gets scored - eval/resolve.py
                        looks up the CURRENT parent_id for each pair against
                        the live corpus every time eval runs. A few entries
                        add "contains": "some substring" - only needed when
                        the same heading repeats across a page with
                        genuinely different content each time (several
                        multi-chapter guides here repeat headings like
                        "Eligibility" once per chapter), to pick out the
                        specific occurrence that actually answers the
                        question rather than accepting any of them.
    must_mention        short phrases a faithful answer should contain
    notes               why the question is here and what it is meant to catch

Why url + heading instead of a parent_id like "employment/prove-right-to-
work#3": a parent_id encodes two things that drift independently of the
actual content - which category folder fetch.py happened to file the page
under this run, and the position chunk.py happened to number it at. Neither
says anything about whether the content is right. A page moving from
employment/ to tax_ni/, or gaining a new intro paragraph that bumps every
later section up by one index, silently breaks a positional id without a
single word of the real answer changing. url + heading survives both,
because it identifies "this section" by what it actually is, not by where it
currently happens to sit.

What this does NOT fix, and cannot: if the page is genuinely taken down, or a
heading is reworded enough that it no longer matches, resolution fails and
that entry needs a human to look at it - not a cleverer id scheme. That
happened for real on 2026-08-15, when a full recreate of laws/ (see
CLAUDE.md's note on ingestion/fetch.py --recreate) found gov.uk had actually
removed several pages this testset depended on. Three entries below were
repointed to a section that still covers the same topic elsewhere in the
corpus, and one was moved into the "questions the corpus cannot answer"
group below because no replacement exists. Every other entry resolved
automatically against the refreshed corpus with no change needed - that is
the whole point of matching on content identity instead of position.

Why scoring uses sections rather than page URLs: a single page can have dozens
of sections that all share one URL, so scoring by URL gives full marks for
finding any part of the right page. Measured on this very set, page-level
scoring reported no change at all between two pipelines, while the actual
sections returned changed on 17 of 18 questions - including moving the correct
section from third place to first. The metric simply could not see it.

Every section here was checked by reading the actual text, not guessed from
the page title. The per-question notes record what was checked.
"""

TESTSET = [
    # --- visa -------------------------------------------------------------
    {
        "question": "What can I not do on a Student visa?",
        "category": ["visa"],
        "expected_sections": [
            {"url": "https://www.gov.uk/student-visa", "heading": "What you can and cannot do"},
        ],
        "must_mention": ["public funds", "self-employed"],
        "notes": "Direct fact lookup. The only section listing restrictions.",
    },
    {
        "question": "How many hours a week can I work on a Student visa during term time?",
        "category": ["visa"],
        "expected_sections": [
            {"url": "https://www.gov.uk/student-visa", "heading": "What you can and cannot do"},
        ],
        "must_mention": [],
        "notes": (
            "KNOWN GAP IN THE CORPUS: this is the only section that discusses work "
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
        "expected_sections": [
            {"url": "https://www.gov.uk/biometric-residence-permits", "heading": None},
        ],
        "must_mention": ["expired", "eVisa"],
        "notes": "Tests a specific, date-sensitive fact. The 'all BRPs have now expired, replaced by eVisas' sentence is in the page intro, not a subsection.",
    },
    {
        "question": "I'm on a Student visa - can I switch to a Skilled Worker visa after I graduate?",
        "category": ["visa", "employment"],
        "expected_sections": [
            {"url": "https://www.gov.uk/skilled-worker-visa", "heading": "Switch to this visa"},
            {"url": "https://www.gov.uk/skilled-worker-visa", "heading": "Eligibility",
             "contains": "Student visa"},
        ],
        "must_mention": [],
        "notes": (
            "Needs two hops: the switching rules live on the SKILLED WORKER page "
            "('Switch to this visa' and 'Eligibility', which explicitly "
            "covers switching from a Student visa), not on the student-visa page "
            "the question sounds like it is about.\n"
            "'Eligibility' is not a unique heading on this page - it appears in "
            "every chapter of the guide (apply from outside the UK, extend this "
            "visa, switch to this visa...), and those sections are NOT "
            "interchangeable. `contains` picks out specifically the switch-to-"
            "this-visa chapter's Eligibility, the only one that mentions Student "
            "visas at all."
        ),
    },

    # --- tax_ni -------------------------------------------------------------
    {
        "question": "Who is eligible to apply for a National Insurance number?",
        "category": ["tax_ni"],
        "expected_sections": [
            {"url": "https://www.gov.uk/apply-national-insurance-number",
             "heading": "Who can apply for a National Insurance number"},
        ],
        "must_mention": ["right to work"],
        "notes": "Direct fact lookup, 'Who can apply' section.",
    },
    {
        "question": "Can I start working before I get my National Insurance number?",
        "category": ["tax_ni"],
        "expected_sections": [
            {"url": "https://www.gov.uk/apply-national-insurance-number",
             "heading": "Who can apply for a National Insurance number"},
        ],
        "must_mention": ["right to work"],
        "notes": (
            "Same section as above, not just the same page - verified the "
            "exact sentence ('You can start work before you receive your National "
            "Insurance number if you can prove you have the right to work') sits "
            "there alongside the eligibility list. Tests it isn't a fluke match."
        ),
    },
    {
        "question": "What is the Personal Allowance for Income Tax?",
        "category": ["tax_ni"],
        "expected_sections": [
            {"url": "https://www.gov.uk/income-tax-rates", "heading": "Your tax-free Personal Allowance"},
        ],
        "must_mention": [],
        "notes": "Numeric fact; 'Your tax-free Personal Allowance' section, not the rates-and-bands table.",
    },

    # --- housing -------------------------------------------------------------
    {
        "question": "Do full-time students have to pay Council Tax?",
        "category": ["housing"],
        "expected_sections": [
            {"url": "https://www.gov.uk/council-tax", "heading": "Discounts for full-time students"},
        ],
        "must_mention": ["full-time student"],
        "notes": "Confirmed working well in manual testing. 'Discounts for full-time students' section specifically.",
    },
    {
        "question": "Does my landlord have to protect my deposit?",
        "category": ["housing"],
        "expected_sections": [
            {"url": "https://www.gov.uk/tenancy-deposit-protection", "heading": "Overview"},
            {"url": "https://www.gov.uk/private-renting", "heading": "Deposit protection"},
        ],
        "must_mention": ["deposit protection scheme"],
        "notes": (
            "Direct fact lookup. TWO sections genuinely answer this and both "
            "count: tenancy-deposit-protection's Overview ('Your landlord must put your "
            "deposit in a government-approved tenancy deposit scheme') and "
            "private-renting's Deposit protection section ('In England, your landlord must keep your "
            "deposit safe using a government-approved tenancy deposit "
            "protection scheme'). The private-renting section was added 2026-08-05 "
            "after it was scored a miss despite materially answering the "
            "question - a grounding bug, not a retrieval failure.\n"
            "This question is also the corpus's best trap for score-based "
            "relevance: tenancy-deposit-protection's 'Holding deposits' section says "
            "the landlord does NOT have to protect a holding deposit, and the "
            "cross-encoder scores it 9.34 - near the highest in the testset - "
            "while it answers the opposite question. See agent/grade.py."
        ),
    },
    {
        "question": "What types of tenancy exist for private renting in the UK?",
        "category": ["housing"],
        "expected_sections": [
            {"url": "https://www.gov.uk/private-renting", "heading": "Tenancy types"},
        ],
        "must_mention": ["tenancy"],
        "notes": "Overview-style question - 'Tenancy types' section on a 47-section page, exactly the kind of within-page distinction page-level scoring couldn't see.",
    },

    # --- employment -------------------------------------------------------------
    {
        "question": "What is the National Minimum Wage and does age affect it?",
        "category": ["employment"],
        "expected_sections": [
            {"url": "https://www.gov.uk/national-minimum-wage-rates", "heading": None},
            {"url": "https://www.gov.uk/national-minimum-wage-rates", "heading": "Current rates"},
        ],
        "must_mention": ["age"],
        "notes": "Split across two parents: the intro states the rate depends on age, 'Current rates' has the actual age-banded rate table. Either answers it.",
    },
    {
        "question": "How many weeks of paid holiday am I entitled to as a worker?",
        "category": ["employment"],
        "expected_sections": [
            {"url": "https://www.gov.uk/holiday-entitlement-rights", "heading": "Entitlement"},
        ],
        "must_mention": ["5.6 weeks"],
        "notes": "Exact-number fact - good BM25/hybrid-search comparison case. 'Entitlement' section, not the pay-calculation detail further down the page.",
    },

    # --- education -------------------------------------------------------------
    {
        "question": "What can I use student finance to help pay for?",
        "category": ["education"],
        "expected_sections": [
            {"url": "https://www.gov.uk/student-finance", "heading": "Overview"},
        ],
        "must_mention": [],
        "notes": "Overview-style question - the page Overview lists Tuition Fee Loan / Maintenance Loan / what each covers.",
    },
    {
        "question": "Can I get student finance if I've studied a degree before?",
        "category": ["education"],
        "expected_sections": [
            {"url": "https://www.gov.uk/student-finance", "heading": "If you've studied before"},
        ],
        "must_mention": [],
        "notes": (
            "This section is one that chunk.py splits "
            "into three overlapping chunks, so this checks the split did not "
            "break retrieval of the fact. It also shows section-level scoring "
            "working: reranking moves this from third place to first."
        ),
    },

    # --- banking -------------------------------------------------------------
    {
        "question": "What is Attendance Allowance and who is it for?",
        "category": ["banking"],
        "expected_sections": [
            {"url": "https://www.gov.uk/attendance-allowance", "heading": "Overview"},
            {"url": "https://www.gov.uk/attendance-allowance", "heading": "Eligibility",
             "contains": "physical disability"},
        ],
        "must_mention": ["disability"],
        "notes": (
            "Two-part question, two-part answer: 'Overview' (what it is) + 'Eligibility' "
            "(who it's for). Also contains a rate table, good structure test.\n"
            "'Eligibility' is not unique on this page - a second, unrelated "
            "'Eligibility' section covers the special end-of-life rules (12 months "
            "or less to live) further down. `contains` picks out the general "
            "eligibility criteria this question is actually asking about, not "
            "the end-of-life special case."
        ),
    },

    # --- nhs -------------------------------------------------------------
    {
        "question": "How do I register with a GP surgery in England?",
        "category": ["nhs"],
        "expected_sections": [
            {"url": "https://www.nhs.uk/nhs-services/gps/how-to-register-with-a-gp-surgery/", "heading": None},
            {"url": "https://www.nhs.uk/nhs-services/gps/how-to-register-with-a-gp-surgery/",
             "heading": "Register or change GP surgery online or using the NHS App"},
        ],
        "must_mention": ["free"],
        "notes": (
            "From the nhs.uk scrape (fetch_nhs.py), not gov.uk - tests both "
            "source types. 'for free' is in the page intro, not the "
            "how-to steps section - either section satisfies the question."
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
    # Every section below was checked by reading the actual text.
    # =====================================================================

    # --- visa: Graduate route -------------------------------------------------
    {
        "question": "After I finish my course can I stay in the UK and look for a job?",
        "category": ["visa"],
        "expected_sections": [
            {"url": "https://www.gov.uk/graduate-visa", "heading": "Overview"},
            {"url": "https://www.gov.uk/graduate-visa", "heading": "What you can and cannot do"},
        ],
        "must_mention": [],
        "notes": (
            "Colloquial phrasing of the Graduate visa. Never says 'Graduate "
            "visa', so pure keyword matching cannot find it - the system has "
            "to connect 'stay after my course' to the route's name. 'Overview' states "
            "the 18-month permission, 'What you can and cannot do' says you can 'look for work'."
        ),
    },
    {
        "question": "My student visa runs out next month - can I still apply for the graduate route?",
        "category": ["visa"],
        "expected_sections": [
            {"url": "https://www.gov.uk/graduate-visa", "heading": "When to apply"},
        ],
        "must_mention": [],
        "notes": (
            "TRAP - timing constraint. 'When to apply' states: 'You must apply before your Student "
            "visa or Tier 4 (General) student visa expires.' The correct answer "
            "hinges on a deadline, and a system that retrieves the general "
            "'How to apply' section instead will miss it entirely."
        ),
    },
    {
        "question": "How long does a graduate visa last if I have a PhD?",
        "category": ["visa"],
        "expected_sections": [
            {"url": "https://www.gov.uk/graduate-visa", "heading": "How long you can stay"},
        ],
        "must_mention": [],
        "notes": (
            "Conditional numeric fact: gives 2 years / 18 months by "
            "application date, then 3 years specifically for a doctoral "
            "qualification. Tests retrieval of the qualifying clause, not just "
            "the headline number."
        ),
    },

    # --- nhs: immigration health surcharge ------------------------------------
    # REPOINTED 2026-08-15: the original source, healthcare-immigration-
    # application, was removed from gov.uk entirely (confirmed via two
    # independent --recreate fetches, both returning a genuine 404 - see
    # CLAUDE.md). No gov.uk page covers this topic any more. The NHS site
    # still does, on the "moving to England from outside the EEA" page's
    # "Immigration health surcharge" section, which substantively covers all
    # three questions below: who pays, how much, and when free NHS access
    # starts. Re-verified by reading the actual current text, not assumed.
    {
        "question": "Do I have to pay something for the NHS when I apply for my visa?",
        "category": ["nhs", "visa"],
        "expected_sections": [
            {"url": "https://www.nhs.uk/nhs-services/visiting-or-moving-to-england/moving-to-england-from-outside-the-european-economic-area-eea/",
             "heading": "Immigration health surcharge"},
        ],
        "must_mention": [],
        "notes": "Student never says 'immigration health surcharge' or 'IHS' - the vocabulary gap is the whole test.",
    },
    {
        "question": "How much is the health surcharge per year for a student?",
        "category": ["nhs", "visa"],
        "expected_sections": [
            {"url": "https://www.nhs.uk/nhs-services/visiting-or-moving-to-england/moving-to-england-from-outside-the-european-economic-area-eea/",
             "heading": "Immigration health surcharge"},
        ],
        "must_mention": [],
        "notes": (
            "Exact-number fact (GBP 776 per year for students) - the section states it directly. "
            "Note: the stored text has a character-encoding bug where '£' renders as "
            "the unicode replacement character; a real ingestion issue, separate from this testset."
        ),
    },
    {
        "question": "When can I actually start using the NHS for free after I arrive?",
        "category": ["nhs", "visa"],
        "expected_sections": [
            {"url": "https://www.nhs.uk/nhs-services/visiting-or-moving-to-england/moving-to-england-from-outside-the-european-economic-area-eea/",
             "heading": "Immigration health surcharge"},
        ],
        "must_mention": [],
        "notes": "States free NHS access 'will apply from the date your visa is granted until it expires' - a timing answer, tested on a page dominated by cost information.",
    },

    # --- employment -----------------------------------------------------------
    {
        "question": "My new boss wants proof I'm allowed to work here - what do I show them?",
        "category": ["employment"],
        "expected_sections": [
            {"url": "https://www.gov.uk/prove-right-to-work", "heading": "If you're not a British or Irish citizen"},
        ],
        "must_mention": [],
        "notes": (
            "Colloquial. Covers non-British/Irish citizens: a share code or eligible "
            "immigration documents. The relevant branch for an international student. "
            "REPOINTED 2026-08-15: page moved from employment/ to tax_ni/ on refetch "
            "(a filing change, not a content change - same URL, same text, verified "
            "by reading it) and the site added several new sub-sections ahead of "
            "this one, which is exactly the position-drift this testset's url+heading "
            "matching exists to survive without needing a manual fix like this note."
        ),
    },
    {
        "question": "Can my employer take money out of my wages?",
        "category": ["employment"],
        "expected_sections": [
            {"url": "https://www.gov.uk/understanding-your-pay", "heading": "Deductions from your pay"},
        ],
        "must_mention": [],
        "notes": "'Deductions from your pay' - lists when deductions are allowed. Plain-language phrasing of a section titled with jargon ('deductions').",
    },
    {
        "question": "I work part time - should I get the same hourly rate as full time staff?",
        "category": ["employment"],
        "expected_sections": [
            {"url": "https://www.gov.uk/understanding-your-pay", "heading": "Part-time workers"},
        ],
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
        "expected_sections": [
            {"url": "https://www.gov.uk/repaying-your-student-loan", "heading": "When you start repaying"},
        ],
        "must_mention": [],
        "notes": (
            "'When you start repaying' - income over your plan's "
            "threshold. Hard because the same page has several near-identical "
            "plan-specific sections all containing threshold "
            "numbers; the general answer must beat them."
        ),
    },
    {
        "question": "Will I get charged extra if I pay my student loan off early?",
        "category": ["tax_ni"],
        "expected_sections": [
            {"url": "https://www.gov.uk/repaying-your-student-loan", "heading": "Early repayments"},
        ],
        "must_mention": [],
        "notes": (
            "NEGATION test. A single sentence: 'There's no penalty for "
            "paying some or all of your loan off early.' The answer is 'no', "
            "and a system that retrieves the interest sections instead will "
            "produce a confidently wrong answer."
        ),
    },
    {
        "question": "I'm going home for 6 months - do I need to tell anyone about my student loan?",
        "category": ["tax_ni"],
        "expected_sections": [
            {"url": "https://www.gov.uk/repaying-your-student-loan", "heading": "If you leave the UK for more than 3 months"},
        ],
        "must_mention": [],
        "notes": "Must tell the Student Loans Company if leaving the UK for more than 3 months. Tests matching '6 months' against a 'more than 3 months' rule.",
    },

    # --- housing --------------------------------------------------------------
    {
        "question": "If I rent out my spare room, do I have to protect my lodger's deposit?",
        "category": ["housing"],
        "expected_sections": [
            {"url": "https://www.gov.uk/rent-room-in-your-home", "heading": "Deposits"},
        ],
        "must_mention": [],
        "notes": (
            "THE KEY TRAP IN THIS SET. Near-identical vocabulary to 'Does my "
            "landlord have to protect my deposit?' but the OPPOSITE answer: "
            "'Resident landlords are not legally required to protect "
            "tenants' deposit'. Any system that matches on deposit-protection "
            "wording alone will retrieve the tenancy-deposit-protection page "
            "and answer exactly backwards. This is the same failure shape as "
            "the 'Holding deposits' case that motivated agent/grade.py."
        ),
    },
    {
        "question": "I've got a lodger living with me - who pays the council tax?",
        "category": ["housing"],
        "expected_sections": [
            {"url": "https://www.gov.uk/rent-room-in-your-home", "heading": "Council Tax"},
        ],
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
        "expected_sections": [
            {"url": "https://www.gov.uk/private-renting-evictions", "heading": "Rules your landlord must follow"},
        ],
        "must_mention": [],
        "notes": "'Rules your landlord must follow' - strict procedures, illegal eviction if not followed. Colloquial phrasing of an eviction-procedure question.",
    },

    # --- visa: TB test + border ------------------------------------------------
    {
        "question": "Do I need a TB test before I come to the UK to study?",
        "category": ["visa"],
        "expected_sections": [
            {"url": "https://www.gov.uk/tb-test-visa", "heading": "Check if you need a TB test for your visa application"},
        ],
        "must_mention": [],
        "notes": "Gives the conditions (6 months or more + lived in a listed country). Page is mostly A-Z country lists, which are strong lexical distractors.",
    },
    {
        "question": "Who is exempt from the TB test?",
        "category": ["visa"],
        "expected_sections": [
            {"url": "https://www.gov.uk/tb-test-visa", "heading": "Who does not need to be tested"},
        ],
        "must_mention": [],
        "notes": (
            "Uses 'exempt', a word that does NOT appear in the target section "
            "(headed 'Who does not need to be tested'). Pure synonym "
            "test - BM25 cannot bridge it, so this isolates the semantic half "
            "of hybrid search."
        ),
    },
    {
        "question": "What happens at UK border control when I arrive?",
        "category": ["visa"],
        "expected_sections": [
            {"url": "https://www.gov.uk/uk-border-control", "heading": "At border control"},
            {"url": "https://www.gov.uk/uk-border-control",
             "heading": "If you're from outside the EU, Switzerland, Norway, Iceland or Liechtenstein"},
        ],
        "must_mention": [],
        "notes": "General border control steps, plus the non-EU route relevant to most international students. Either answers it.",
    },

    # --- education -------------------------------------------------------------
    {
        "question": "Can I get help paying for childcare while I'm studying?",
        "category": ["education"],
        "expected_sections": [
            {"url": "https://www.gov.uk/childcare-grant", "heading": "Overview"},
            {"url": "https://www.gov.uk/childcare-grant", "heading": "Eligibility"},
        ],
        "must_mention": [],
        "notes": (
            "Overview / Eligibility. Adds coverage to `education`, the thinnest category in the corpus. "
            "REPOINTED 2026-08-15: the page was restructured on refetch (a real content "
            "change, verified by reading it - see the childcare-grant diff noted in "
            "CLAUDE.md), which moved Eligibility to a different position. Content "
            "and topic are unchanged, only where chunk.py numbered it."
        ),
    },
    {
        "question": "How much childcare grant can I get for two kids?",
        "category": ["education"],
        "expected_sections": [
            {"url": "https://www.gov.uk/childcare-grant", "heading": "What you'll get"},
        ],
        "must_mention": [],
        "notes": (
            "REPOINTED 2026-08-15: this was previously an AMBIGUITY test, since the page held "
            "two near-identical rate sections for different academic years. The "
            "2026-08-15 refetch found gov.uk had genuinely restructured the page "
            "to state only the current rates in one 'What you'll get' section - "
            "an actual content simplification upstream, not a chunking artifact. "
            "The ambiguity this question was designed to test no longer exists "
            "on the live page, so this is now a plain numeric-fact lookup instead."
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
        "expected_sections": [],
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
        "expected_sections": [],
        "must_mention": [],
        "notes": (
            "Checked: not a single chunk in the corpus mentions university "
            "rankings or league tables. Unlike the pet-dog question this is a "
            "very plausible student question - it is just about something "
            "gov.uk does not publish an opinion on, so the only honest answer "
            "is to say so."
        ),
    },
    {
        "question": "What does the NHS entitlements migrant health guide cover?",
        "category": ["nhs"],
        "expected_sections": [],
        "must_mention": [],
        "notes": (
            "MOVED HERE 2026-08-15: the source page, gov.uk/guidance/nhs-"
            "entitlements-migrant-health-guide, was removed from gov.uk "
            "entirely - confirmed via two independent --recreate fetches, "
            "both returning a genuine 404 (see CLAUDE.md). Checked for a "
            "replacement: no other page in the corpus describes what this "
            "specific guide covers as a document. Previously this was a "
            "direct fact lookup, deliberately left as a genuine miss because "
            "retrieval returned real content FROM the guide without matching "
            "what the guide's own purpose section said. Now the guide itself "
            "is gone, so declining is the only honest answer - the same "
            "reasoning as the pet-dog and university-ranking questions above, "
            "just arrived at by the source disappearing rather than never "
            "existing."
        ),
    },
]


if __name__ == "__main__":
    from collections import Counter
    cats = Counter(c for q in TESTSET for c in q["category"])
    print(f"{len(TESTSET)} questions")
    for cat, n in cats.most_common():
        print(f"  {cat:12} {n}")
