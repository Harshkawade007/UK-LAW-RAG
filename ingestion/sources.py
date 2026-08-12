"""
The list of pages to start downloading from. No code here, just data.

SOURCES holds gov.uk PATHS rather than full URLs, because fetch.py turns each
one into an API call: https://www.gov.uk/api/content/<path>

NHS_SOURCES holds full nhs.uk URLs, because fetch_nhs.py fetches those pages
directly.

These are only STARTING points. Running the fetchers with --discover follows
the links on each page and grows the collection well beyond this list.

Run this file on its own to see how many seeds there are per category:

    python sources.py
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
# NHS pages. These need fetch_nhs.py rather than fetch.py, because nhs.uk is an
# ordinary website with no API - so its pages have to be read from HTML.
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

# Only follow links that start with one of these. This keeps the crawl on "how
# to use the NHS" pages and out of the enormous A-Z of medical conditions,
# which is not what this project is about.
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
