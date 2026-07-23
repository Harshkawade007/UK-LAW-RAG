"""
sources.py

Hand-picked seed pages for each law category.
These are gov.uk PATHS (not full URLs) - fetch.py turns them into
API calls like https://www.gov.uk/api/content/<path>

These seeds are just the STARTING points. Run `python fetch.py --discover`
to follow each page's related links and auto-grow the corpus well beyond
this list. Re-running is safe - it won't re-fetch what you already have.
"""

SOURCES = {
    "visa": [
        "student-visa",
        "student-visa/family-members",
        "student-visa/extend-your-visa",
        "student-visa/switch-to-this-visa",
        "student-visa/knowledge-of-english",
        "child-study-visa",
        "graduate-visa",
        "skilled-worker-visa",
        "temporary-worker-government-authorised-exchange-visa",
        "standard-visitor",
        "biometric-residence-permits",
        "prove-right-to-live-in-the-uk",
        "view-prove-immigration-status",
        "apply-to-come-to-the-uk",
        "tier-4-general-visa",  # legacy route, may redirect
        "immigration-health-surcharge-for-visa-and-immigration-applications",
        "guidance/immigration-rules",
    ],
    "tax_ni": [
        "apply-national-insurance-number",
        "national-insurance",
        "national-insurance/how-much-you-pay",
        "national-insurance-rates-letters",
        "income-tax",
        "income-tax-rates",
        "income-tax/how-you-pay-income-tax",
        "tax-come-to-uk",
        "tax-uk-income-live-abroad",
        "paye-forms-p45-p60-p11d",
        "claim-tax-refund",
        "check-income-tax-current-year",
        "tax-foreign-income",
        "personal-tax-account",
        "self-assessment-tax-returns",
    ],
    "housing": [
        "council-tax",
        "council-tax/discounts-for-full-time-students",
        "council-tax/who-has-to-pay",
        "private-renting",  # guide - includes the tenancy-agreements chapter
        "private-renting-tenancy-agreements",
        "renting-out-a-property",
        "deposit-protection-schemes-and-landlords",
        "tenancy-deposit-protection",
        "housing-and-local-services",
        "government/publications/how-to-rent",
        "rent-arrears-and-evictions",
        "private-renting-evictions",
        "shelter-housing-advice",  # may redirect / 404, harmless
        "apply-for-council-housing",
    ],
    "banking": [
        # Most student-banking content lives on MoneyHelper, not gov.uk.
        # These are the gov.uk money/debt pages that do exist.
        "debt-advice",
        "budgeting-help-benefits",
        "financial-help-disabled",
        "government/collections/basic-bank-accounts",
    ],
    "nhs": [
        "healthcare-immigration-application",
        "pay-for-uk-healthcare-abroad",
        "guidance/nhs-entitlements-migrant-health-guide",
        "government/publications/nhs-entitlements-migrant-health-guide",
        # register-with-a-gp-surgery lives on nhs.uk (separate site) - not fetchable here.
    ],
    "employment": [
        "minimum-wage-different-types-work",
        "national-minimum-wage-rates",
        "minimum-wage-for-different-types-of-work",
        "employment-contracts-and-conditions",
        "contract-types-and-employer-responsibilities",
        "employment-status",
        "prove-right-to-work",  # was "rights-workplace-immigration-status" (404)
        "your-right-to-minimum-wage",
        "maximum-weekly-working-hours",
        "holiday-entitlement-rights",
        "being-monitored-at-work",
        "national-insurance/what-national-insurance-is-for",
        "pay-and-work-rights-helpline",
        "understanding-your-pay",
        "payslips",
        "sick-leave-pay-employees",
    ],
    "education": [
        "student-finance",
        "student-finance-register-login",
        "apply-online-for-student-finance",
        "further-education-courses-qualifications",
        "advanced-learner-loan",
        "extra-money-pay-university",
        "apply-for-student-finance",
        "student-visa/money",
    ],
}

# ---------------------------------------------------------------------------
# NHS sources.
# NHS content lives on nhs.uk (a plain website, no Content API), so it needs
# the separate HTML scraper in fetch_nhs.py rather than fetch.py.
# These are full URLs. The scraper crawls related links but stays inside
# NHS_ALLOWED_PREFIXES (below) so it never wanders into the /conditions/
# medical encyclopedia, which isn't relevant to a student admin assistant.
# ---------------------------------------------------------------------------
NHS_SOURCES = [
    "https://www.nhs.uk/nhs-services/gps/how-to-register-with-a-gp-surgery/",
    "https://www.nhs.uk/nhs-services/gps/",
    "https://www.nhs.uk/using-the-nhs/nhs-services/",
    "https://www.nhs.uk/nhs-services/prescriptions/",
    "https://www.nhs.uk/nhs-services/dentists/",
    "https://www.nhs.uk/nhs-services/pharmacies/",
    "https://www.nhs.uk/nhs-services/urgent-and-emergency-care/",
    "https://www.nhs.uk/nhs-services/mental-health-services/",
    "https://www.nhs.uk/using-the-nhs/healthcare-abroad/",
    "https://www.nhs.uk/using-the-nhs/nhs-services/visiting-or-moving-to-england/",
    "https://www.nhs.uk/nhs-app/",
]

# Only crawl links whose path starts with one of these. Keeps the scraper on
# "how to use the NHS" content and out of the clinical conditions A-Z.
NHS_ALLOWED_PREFIXES = (
    "/nhs-services/",
    "/using-the-nhs/",
    "/nhs-app/",
)


if __name__ == "__main__":
    total = sum(len(v) for v in SOURCES.values())
    print(f"{len(SOURCES)} gov.uk categories, {total} seed pages total")
    for category, paths in SOURCES.items():
        print(f"  {category}: {len(paths)} pages")
    print(f"nhs: {len(NHS_SOURCES)} seed URLs")
