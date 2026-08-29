"""Generate the committed demonstration corpus.

The specification requires a corpus with real, machine-detectable structure,
committed to the repository, parsed offline.  A published Act cannot be
redistributed here without dragging a licensing question into a teaching repo,
so this module *generates* one: a synthetic Act laid out exactly as
``pdftotext -layout`` renders a UK Public General Act, including the four traps
the parser has to survive.

    uv run rights-corpus --out data/corpus.layout.txt

Output is a pure function of this file: regenerating twice produces identical
bytes, so ``index_version`` is stable across machines.  To use a real Act
instead, drop the PDF in ``data/`` and point ``RIGHTS_CORPUS`` at it -- the parser
reads either a layout text file or a PDF.

Deliberately included structural traps (see parser docstring):

1. A table of contents that looks exactly like the body.
2. Running headers repeated on every page, in mixed case (``Part 1 — …``) so a
   case-insensitive filter would also swallow the real ``PART 1`` markers.
3. Quoted material that mimics a cross-heading but is followed by indented text.
4. Provisions inserted into *another* Act, numbered with a letter suffix at
   non-zero indent.
"""

from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass
from pathlib import Path

from rights_agent.entrypoints import operator_error_exit

SHORT_TITLE = "Employment Rights (Demonstration) Act 2026"
CHAPTER = "2026 CHAPTER 14"
ROYAL_ASSENT = "28th August 2026"
HOST_ACT = "Employment Rights Act 1996"
PAGE_WIDTH = 80
PAGE_LINES = 52

WORKER = "worker"


@dataclass(frozen=True, slots=True)
class Section:
    title: str
    subject: str
    term_a: str
    term_b: str
    subsections: int = 6


@dataclass(frozen=True, slots=True)
class Theme:
    heading: str
    sections: tuple[Section, ...]


@dataclass(frozen=True, slots=True)
class Part:
    number: int
    title: str
    themes: tuple[Theme, ...]


def _s(title: str, subject: str, a: str, b: str, n: int = 6) -> Section:
    return Section(title, subject, a, b, n)


PARTS: tuple[Part, ...] = (
    Part(
        1,
        "EMPLOYMENT RIGHTS",
        (
            Theme(
                "Zero hours workers, etc",
                (
                    _s("Right to guaranteed hours", "a guaranteed hours offer", "reference period", "qualifying hours", 7),
                    _s("Guaranteed hours: number of hours to be offered", "the number of guaranteed hours", "hours worked", "regular pattern"),
                    _s("Guaranteed hours: exceptions for temporary need", "a temporary work exception", "temporary need", "fixed term"),
                    _s("Duty to give reasonable notice of a shift", "notice of a shift", "shift notice", "reasonable notice", 7),
                    _s("Right to payment for a cancelled shift", "payment for a cancelled shift", "short notice cancellation", "cancellation payment"),
                    _s("Zero hours workers: agency workers", "the application of this Chapter to agency workers", "agency worker", "hirer"),
                ),
            ),
            Theme(
                "Flexible working",
                (
                    _s("Right to request flexible working", "a flexible working request", "flexible working", "working pattern", 7),
                    _s("Flexible working: grounds for refusal", "the grounds on which a request may be refused", "business ground", "reasonableness"),
                    _s("Flexible working: duty to consult", "consultation on a request", "consultation", "alternative pattern"),
                    _s("Flexible working: decision period", "the decision period for a request", "decision period", "extension"),
                    _s("Flexible working: complaints to a tribunal", "complaints about flexible working", "complaint", "declaration"),
                ),
            ),
            Theme(
                "Statutory sick pay",
                (
                    _s("Removal of the waiting period for statutory sick pay", "the waiting period", "waiting period", "period of incapacity"),
                    _s("Statutory sick pay: removal of the lower earnings limit", "the lower earnings limit", "lower earnings limit", "percentage rate"),
                    _s("Statutory sick pay: rate payable", "the rate of statutory sick pay", "weekly rate", "normal weekly earnings"),
                    _s("Statutory sick pay: notification of incapacity", "notification of incapacity for work", "notification", "evidence of incapacity"),
                    _s("Statutory sick pay: records and inspection", "records of statutory sick pay", "record keeping", "inspection"),
                ),
            ),
            Theme(
                "Family leave",
                (
                    _s("Paternity leave: removal of the qualifying period", "the qualifying period for paternity leave", "paternity leave", "continuous employment"),
                    _s("Unpaid parental leave: removal of the qualifying period", "the qualifying period for parental leave", "parental leave", "notice of intention"),
                    _s("Right to bereavement leave", "bereavement leave", "bereaved person", "relevant relationship", 8),
                    _s("Bereavement leave: length and timing", "the length of bereavement leave", "leave period", "window for taking leave"),
                    _s("Bereavement leave: pregnancy loss", "bereavement leave following pregnancy loss", "pregnancy loss", "eligible person"),
                    _s("Protection from detriment: family leave", "detriment connected with family leave", "detriment", "prescribed reason"),
                ),
            ),
            Theme(
                "Dismissal",
                (
                    _s("Removal of the qualifying period for unfair dismissal", "the qualifying period for unfair dismissal", "qualifying period", "initial period of employment", 7),
                    _s("Initial period of employment: modified test", "the modified test during the initial period", "modified test", "light touch procedure"),
                    _s("Dismissal during pregnancy and after childbirth", "protection from dismissal during pregnancy", "protected period", "prescribed circumstances"),
                    _s("Dismissal for refusing a variation of contract", "dismissal for refusing a contractual variation", "restricted variation", "fire and rehire"),
                    _s("Collective consultation: dismissal for refusal", "consultation before dismissal for refusal", "collective consultation", "protective award"),
                ),
            ),
            Theme(
                "Harassment and equality",
                (
                    _s("Duty to prevent sexual harassment", "the duty to take all reasonable steps", "all reasonable steps", "preventative duty", 7),
                    _s("Harassment by third parties", "liability for harassment by third parties", "third party", "course of employment"),
                    _s("Equality action plans", "equality action plans", "action plan", "gender pay gap"),
                    _s("Protected disclosures: sexual harassment", "disclosures about sexual harassment", "protected disclosure", "public interest"),
                    _s("Non-disclosure agreements: void provisions", "provisions purporting to prevent disclosure", "non-disclosure agreement", "void provision"),
                ),
            ),
        ),
    ),
    Part(
        2,
        "PAY, DEDUCTIONS AND WORKING TIME",
        (
            Theme(
                "Written statements and payslips",
                (
                    _s("Written statement of employment particulars", "the written statement of particulars", "written statement", "particulars of employment", 7),
                    _s("Written statement: information about guaranteed hours", "information about guaranteed hours", "hours information", "reference period"),
                    _s("Itemised pay statement: additional information", "itemised pay statements", "pay statement", "itemised deduction"),
                    _s("Statement of variable hours and rates", "statements about variable hours", "variable hours", "rate of pay"),
                    _s("Failure to provide a statement: remedies", "remedies for failure to provide a statement", "remedy", "award"),
                ),
            ),
            Theme(
                "Deductions from wages",
                (
                    _s("Unauthorised deductions from wages", "unauthorised deductions", "deduction", "properly payable", 7),
                    _s("Deductions for training costs", "deductions for training costs", "training cost", "repayment clause"),
                    _s("Deductions for equipment and uniform", "deductions for equipment and uniform", "equipment cost", "uniform"),
                    _s("Deductions: limits on recovery", "limits on the recovery of deductions", "recovery limit", "net wages"),
                    _s("Deductions: complaints and time limits", "complaints about deductions", "time limit", "series of deductions"),
                ),
            ),
            Theme(
                "Allocation of tips",
                (
                    _s("Fair allocation of tips, gratuities and service charges", "the fair allocation of tips", "qualifying tip", "fair allocation", 7),
                    _s("Tips: written policy", "a written policy on tips", "tipping policy", "publication"),
                    _s("Tips: records and access", "records of tips", "tipping record", "right of access"),
                    _s("Tips: agency workers", "the allocation of tips to agency workers", "agency worker", "eligible worker"),
                    _s("Tips: enforcement", "the enforcement of tipping provisions", "enforcement", "revision of allocation"),
                ),
            ),
            Theme(
                "Holiday pay and working time",
                (
                    _s("Calculation of a week's pay for holiday purposes", "the calculation of a week's pay", "week's pay", "reference period", 7),
                    _s("Rolled-up holiday pay: irregular hours workers", "rolled-up holiday pay", "irregular hours worker", "accrual"),
                    _s("Carry-over of untaken annual leave", "the carry-over of annual leave", "carry-over", "untaken leave"),
                    _s("Records of working time", "records of hours worked", "working time record", "retention period"),
                    _s("Right to disconnect: guidance", "guidance about contact outside working hours", "out of hours contact", "guidance"),
                ),
            ),
            Theme(
                "National minimum wage",
                (
                    _s("Single adult rate of the national minimum wage", "the single adult rate", "adult rate", "age band", 7),
                    _s("National minimum wage: accommodation offset", "the accommodation offset", "accommodation offset", "daily amount"),
                    _s("National minimum wage: travel time", "travel time and the minimum wage", "travel time", "assignment"),
                    _s("National minimum wage: sleep-in shifts", "sleep-in shifts", "sleep-in shift", "availability for work"),
                    _s("National minimum wage: notices of underpayment", "notices of underpayment", "notice of underpayment", "arrears"),
                ),
            ),
            Theme(
                "Umbrella companies and intermediaries",
                (
                    _s("Regulation of umbrella companies", "the regulation of umbrella companies", "umbrella company", "intermediary", 7),
                    _s("Umbrella companies: key information documents", "key information documents", "key information document", "assignment rate"),
                    _s("Joint and several liability for unpaid wages", "joint and several liability", "joint liability", "unpaid wages"),
                    _s("Intermediaries: transparency of deductions", "transparency of deductions by intermediaries", "transparency", "margin"),
                    _s("Intermediaries: enforcement powers", "enforcement against intermediaries", "enforcement power", "labour supply chain"),
                ),
            ),
        ),
    ),
    Part(
        3,
        "PROTECTION FROM DISMISSAL AND DETRIMENT",
        (
            Theme(
                "Unfair dismissal",
                (
                    _s("Right not to be unfairly dismissed", "the right not to be unfairly dismissed", "unfair dismissal", "reason for dismissal", 8),
                    _s("Fairness of a dismissal: procedure", "the procedure for a fair dismissal", "fair procedure", "reasonable investigation"),
                    _s("Dismissal for a substantial business reason", "dismissal for a business reason", "business reason", "substantial reason"),
                    _s("Automatically unfair reasons for dismissal", "automatically unfair reasons", "automatically unfair", "prescribed reason"),
                    _s("Written reasons for dismissal", "written reasons for dismissal", "written reasons", "request"),
                ),
            ),
            Theme(
                "Remedies for unfair dismissal",
                (
                    _s("Interim relief in dismissal proceedings", "interim relief", "interim relief", "continuation of contract", 7),
                    _s("Basic award and compensatory award", "the basic and compensatory awards", "basic award", "compensatory award"),
                    _s("Reduction of an award for contributory conduct", "reduction for contributory conduct", "contributory conduct", "just and equitable"),
                    _s("Reinstatement and re-engagement", "reinstatement and re-engagement", "reinstatement", "re-engagement"),
                    _s("Time limits for dismissal complaints", "time limits for complaints", "time limit", "extension of time"),
                ),
            ),
            Theme(
                "Variation of contract",
                (
                    _s("Restricted variations of contract", "restricted variations", "restricted variation", "consultation requirement", 7),
                    _s("Dismissal to effect a restricted variation", "dismissal to effect a variation", "variation dismissal", "financial difficulties"),
                    _s("Restricted variations: evidence of financial difficulties", "evidence of financial difficulties", "financial difficulty", "evidence"),
                    _s("Restricted variations: guidance", "guidance about restricted variations", "guidance", "code of practice"),
                    _s("Restricted variations: interaction with collective agreements", "collective agreements and variations", "collective agreement", "recognised union"),
                ),
            ),
            Theme(
                "Detriment and whistleblowing",
                (
                    _s("Protection from detriment for exercising a right", "detriment for exercising a right", "detriment", "prescribed right", 7),
                    _s("Protected disclosures: qualifying disclosures", "qualifying disclosures", "qualifying disclosure", "relevant failure"),
                    _s("Protected disclosures: prescribed persons", "disclosures to prescribed persons", "prescribed person", "reasonable belief"),
                    _s("Detriment: burden of proof", "the burden of proof in detriment claims", "burden of proof", "explanation"),
                    _s("Detriment: compensation", "compensation for detriment", "compensation", "injury to feelings"),
                ),
            ),
            Theme(
                "Redundancy",
                (
                    _s("Collective redundancies: threshold for consultation", "the threshold for collective consultation", "consultation threshold", "establishment", 7),
                    _s("Collective redundancies: consultation period", "the consultation period", "consultation period", "minimum period"),
                    _s("Protective awards for failure to consult", "protective awards", "protective award", "period of the award"),
                    _s("Redundancy: suitable alternative employment", "suitable alternative employment", "suitable alternative", "trial period"),
                    _s("Redundancy payments: calculation", "the calculation of redundancy payments", "redundancy payment", "week's pay"),
                ),
            ),
            Theme(
                "Continuity of employment",
                (
                    _s("Continuous employment: computation", "the computation of continuous employment", "continuous employment", "week counting", 7),
                    _s("Continuity across a change of employer", "continuity on a change of employer", "change of employer", "associated employer"),
                    _s("Continuity: industrial action and lock-outs", "continuity during industrial action", "industrial action", "lock-out"),
                    _s("Continuity: reinstatement after appeal", "continuity on reinstatement", "reinstatement", "internal appeal"),
                    _s("Continuity: presumptions", "presumptions about continuity", "presumption", "evidence"),
                ),
            ),
        ),
    ),
    Part(
        4,
        "TRADE UNIONS AND INDUSTRIAL RELATIONS",
        (
            Theme(
                "Trade union recognition",
                (
                    _s("Application for recognition", "applications for trade union recognition", "recognition application", "bargaining unit", 7),
                    _s("Recognition: admissibility of an application", "the admissibility of an application", "admissibility", "membership threshold"),
                    _s("Recognition: determination of the bargaining unit", "determination of the bargaining unit", "bargaining unit", "effective management"),
                    _s("Recognition: ballots", "recognition ballots", "recognition ballot", "majority support"),
                    _s("Recognition: method of collective bargaining", "the method of collective bargaining", "bargaining method", "specified method"),
                ),
            ),
            Theme(
                "Union access and facilities",
                (
                    _s("Right of access to workplaces", "the right of access to a workplace", "access agreement", "access request", 7),
                    _s("Access agreements: terms and conditions", "the terms of an access agreement", "access terms", "reasonable conditions"),
                    _s("Access: digital communications", "access by digital means", "digital access", "communication"),
                    _s("Facilities for union representatives", "facilities for representatives", "facility", "accredited representative"),
                    _s("Time off for union duties and training", "time off for union duties", "time off", "union duty"),
                ),
            ),
            Theme(
                "Industrial action",
                (
                    _s("Notice of industrial action", "notice of industrial action to an employer", "notice period", "industrial action", 7),
                    _s("Industrial action ballots: turnout requirements", "turnout requirements for ballots", "turnout", "ballot threshold"),
                    _s("Electronic balloting", "electronic balloting", "electronic ballot", "secrecy"),
                    _s("Protection from detriment for industrial action", "detriment for taking industrial action", "detriment", "protected action"),
                    _s("Duration of a mandate for industrial action", "the duration of a ballot mandate", "mandate", "expiry"),
                ),
            ),
            Theme(
                "Blacklisting and information",
                (
                    _s("Prohibition of blacklisting", "the prohibition of blacklists", "blacklist", "prohibited list", 7),
                    _s("Blacklisting: compensation", "compensation for blacklisting", "compensation", "minimum award"),
                    _s("Information and consultation agreements", "information and consultation agreements", "consultation agreement", "employee request"),
                    _s("Information about the workforce", "information about the workforce", "workforce information", "aggregate data"),
                    _s("Disclosure of information for collective bargaining", "disclosure for collective bargaining", "disclosure", "bargaining purpose"),
                ),
            ),
            Theme(
                "School support staff and social care",
                (
                    _s("School Support Staff Negotiating Body", "the School Support Staff Negotiating Body", "negotiating body", "support staff", 7),
                    _s("Negotiating body: remit", "the remit of the negotiating body", "remit", "terms and conditions"),
                    _s("Adult Social Care Negotiating Body", "the Adult Social Care Negotiating Body", "social care", "sector agreement"),
                    _s("Sector agreements: ratification", "the ratification of a sector agreement", "ratification", "approval"),
                    _s("Sector agreements: effect on contracts", "the effect of a sector agreement", "contractual effect", "incorporation"),
                ),
            ),
            Theme(
                "Union administration",
                (
                    _s("Union membership records", "union membership records", "membership record", "assurance"),
                    _s("Political funds and opt-in", "political funds", "political fund", "opt-in notice"),
                    _s("Deduction of union subscriptions", "the deduction of subscriptions", "check-off", "subscription"),
                    _s("Certification Officer: functions", "the functions of the Certification Officer", "Certification Officer", "investigation"),
                    _s("Certification Officer: levy", "the levy payable to the Certification Officer", "levy", "prescribed amount"),
                ),
            ),
        ),
    ),
    Part(
        5,
        "ENFORCEMENT",
        (
            Theme(
                "The Fair Work Agency",
                (
                    _s("Establishment of the Fair Work Agency", "the establishment of the Agency", "the Agency", "labour market enforcement", 7),
                    _s("Agency: general objective", "the general objective of the Agency", "general objective", "compliance"),
                    _s("Agency: annual strategy and report", "the annual strategy", "annual strategy", "report"),
                    _s("Agency: advisory board", "the advisory board", "advisory board", "membership"),
                    _s("Transfer of enforcement functions to the Agency", "the transfer of functions", "transfer scheme", "existing function"),
                ),
            ),
            Theme(
                "Enforcement officers",
                (
                    _s("Appointment of enforcement officers", "the appointment of enforcement officers", "enforcement officer", "warrant", 7),
                    _s("Powers of entry and inspection", "powers of entry", "power of entry", "premises"),
                    _s("Power to require information", "the power to require information", "information notice", "specified period"),
                    _s("Power to take copies and remove documents", "the power to take copies", "copy", "removal"),
                    _s("Obstruction of an enforcement officer", "obstruction of an officer", "obstruction", "offence"),
                ),
            ),
            Theme(
                "Notices and penalties",
                (
                    _s("Compliance notices", "compliance notices", "compliance notice", "specified steps", 7),
                    _s("Compliance notices: appeals", "appeals against a compliance notice", "appeal", "grounds of appeal"),
                    _s("Penalty notices", "penalty notices", "penalty notice", "penalty amount"),
                    _s("Penalty notices: calculation of the penalty", "the calculation of a penalty", "calculation", "maximum penalty"),
                    _s("Labour market enforcement undertakings", "enforcement undertakings", "undertaking", "prohibited conduct"),
                    _s("Labour market enforcement orders", "enforcement orders", "enforcement order", "breach"),
                ),
            ),
            Theme(
                "Recovery of sums and offences",
                (
                    _s("Recovery of sums due to a worker", "the recovery of sums due", "recoverable sum", "assessment", 7),
                    _s("Recovery: interest and administrative costs", "interest on recoverable sums", "interest", "administrative cost"),
                    _s("Offences by bodies corporate", "offences by bodies corporate", "body corporate", "officer liability"),
                    _s("Time limits for proceedings", "time limits for proceedings", "time limit", "summary offence"),
                    _s("Sentencing and fines", "sentencing for offences", "fine", "statutory maximum"),
                ),
            ),
            Theme(
                "Information sharing",
                (
                    _s("Sharing of information between authorities", "the sharing of information", "information sharing", "relevant authority", 7),
                    _s("Information sharing: data protection", "data protection and information sharing", "data protection", "lawful basis"),
                    _s("Disclosure to the Agency", "disclosure of information to the Agency", "disclosure", "prescribed body"),
                    _s("Restrictions on onward disclosure", "restrictions on onward disclosure", "onward disclosure", "consent"),
                    _s("Publication of enforcement outcomes", "the publication of enforcement outcomes", "publication", "naming scheme"),
                ),
            ),
            Theme(
                "Employment tribunal procedure",
                (
                    _s("Extension of time limits for tribunal claims", "the extension of time limits", "time limit", "six months", 7),
                    _s("Tribunal awards: uplift for breach of a code", "uplift for breach of a code of practice", "uplift", "code of practice"),
                    _s("Costs and deposit orders", "costs and deposit orders", "costs order", "deposit"),
                    _s("Representation and assistance", "representation before a tribunal", "representation", "assistance"),
                    _s("Awards: enforcement of tribunal awards", "the enforcement of awards", "enforcement of award", "penalty for non-payment"),
                ),
            ),
        ),
    ),
    Part(
        6,
        "GENERAL",
        (
            Theme(
                "Regulations and guidance",
                (
                    _s("Power to make regulations", "the power to make regulations", "regulation", "procedure", 7),
                    _s("Regulations: consultation requirements", "consultation before making regulations", "consultation", "representative body"),
                    _s("Codes of practice", "codes of practice", "code of practice", "revision"),
                    _s("Guidance: duty to have regard", "the duty to have regard to guidance", "guidance", "duty to have regard"),
                    _s("Consequential and transitional provision", "consequential provision", "consequential amendment", "transitional provision"),
                ),
            ),
            Theme(
                "Interpretation",
                (
                    _s("General interpretation", "the interpretation of this Act", "interpretation", "defined term", 8),
                    _s("Meaning of employer and worker", "the meaning of employer and worker", "employer", "worker"),
                    _s("Meaning of a week's pay", "the meaning of a week's pay", "week's pay", "calculation date"),
                    _s("Meaning of relevant reference period", "the meaning of the relevant reference period", "reference period", "prescribed period"),
                    _s("Index of defined expressions", "the index of defined expressions", "index", "expression"),
                ),
            ),
            Theme(
                "Final provisions",
                (
                    _s("Financial provision", "financial provision", "expenditure", "money provided by Parliament", 4),
                    _s("Extent", "the extent of this Act", "extent", "Northern Ireland", 5),
                    _s("Commencement", "commencement", "commencement regulations", "appointed day", 6),
                    _s("Short title", "the short title", "short title", "citation", 3),
                ),
            ),
        ),
    ),
)

#: The substantive duty of a section: always the first subsection, so a section
#: read on its own says what it does.
PRIMARY_TEMPLATES: tuple[str, ...] = (
    "An employer must, in relation to each {worker} to whom this section applies, secure {subject} in accordance with regulations under this section.",
    "A {worker} has the right to {subject} where the conditions in subsection ({next}) are met.",
    "This section applies where an employer proposes to make a decision about {subject} which affects a {worker}.",
    "An employer must not, by reason of {term_a}, treat a {worker} less favourably in relation to {subject}.",
    "The Secretary of State must by regulations make provision about {subject}, including provision about {term_a} and {term_b}.",
    "Where a {worker} makes a request relating to {subject}, the employer must deal with the request in accordance with this section.",
)

SECONDARY_TEMPLATES: tuple[str, ...] = (
    "The employer must give the {worker} a written notice setting out {subject} before the end of the period of {days} days beginning with the day on which the duty under subsection (1) arises.",
    'For the purposes of this section, "{term_a}" means the period determined in accordance with regulations, and "{term_b}" is to be construed accordingly.',
    "Subsection ({prev}) does not apply where the employer can show that the requirement was not reasonably practicable having regard to {term_b}.",
    "A {worker} may present a complaint to an employment tribunal that the employer has failed to comply with subsection ({prev}).",
    "Where an employment tribunal finds a complaint under this section well-founded, it must make a declaration to that effect and may order the employer to pay compensation to the {worker}.",
    "The amount of compensation under this section must be such amount as the tribunal considers just and equitable, not exceeding {cap} weeks' pay.",
    "The Secretary of State may by regulations specify the descriptions of {term_a} to be taken into account for the purposes of subsection ({prev}).",
    "Regulations under subsection ({prev}) may make provision about {term_b}, including provision conferring a discretion on an employment tribunal.",
    "Nothing in this section affects any right of the {worker} under a contract of employment, a collective agreement or any other enactment relating to {term_b}.",
    "This section applies whether or not the {worker} has been continuously employed for a period of {months} months ending with the relevant date.",
    "In determining whether an employer has complied with subsection ({prev}), a tribunal must have regard to any relevant code of practice and to the size and administrative resources of the employer's undertaking.",
    "The threshold for the purposes of subsection ({prev}) is £{amount}, or such other amount as may be prescribed.",
    "The employer must keep a record of {subject} for a period of {years} years beginning with the end of the relevant reference period, and must make the record available to the {worker} on request.",
    "Where {term_a} changes during the relevant reference period, the employer must review {subject} and notify the {worker} of the outcome of the review.",
    "An employer who fails to comply with subsection ({prev}) is to be treated for the purposes of Part 5 as having failed to comply with a relevant labour market requirement.",
    "This section does not apply in relation to a {worker} who is employed under a contract for a fixed term of less than {weeks} weeks, unless regulations provide otherwise.",
)

DAYS = (7, 14, 21, 28, 5, 10)
MONTHS = (1, 2, 3, 6, 12, 24)
CAPS = (2, 4, 8, 12, 13, 26)
AMOUNTS = (500, 750, 1000, 1500, 2500, 5000)
YEARS = (2, 3, 6)
WEEKS = (4, 8, 12, 26, 52)

#: Sections that additionally insert provisions into another Act.  Chosen by
#: position so the trap is always exercised: index within the flat section list.
INSERTED_AT: dict[int, tuple[str, str, str]] = {
    0: ("27BA", "Right to guaranteed hours", "guaranteed hours"),
    12: ("64A", "Statutory sick pay: first day of incapacity", "statutory sick pay"),
    22: ("80EA", "Bereavement leave", "bereavement leave"),
    47: ("27JA", "Allocation of qualifying tips", "qualifying tips"),
    73: ("104BA", "Dismissal for refusing a restricted variation", "restricted variations"),
    101: ("70ZA", "Access to workplaces", "workplace access"),
}


def _wrap(text: str, indent: int, hanging: int | None = None) -> list[str]:
    hanging = indent if hanging is None else hanging
    lines = textwrap.wrap(
        text,
        width=PAGE_WIDTH - 4,
        initial_indent=" " * indent,
        subsequent_indent=" " * hanging,
        break_long_words=False,
    )
    return lines or [" " * indent]


def _centre(text: str) -> str:
    pad = max(0, (PAGE_WIDTH - len(text)) // 2)
    return " " * pad + text


@dataclass
class _Cursor:
    """Accumulates body lines and remembers which Part they belong to."""

    lines: list[tuple[str, str]]  # (line, running header for the page)
    part_header: str = ""

    def emit(self, *lines: str) -> None:
        for line in lines:
            self.lines.append((line.rstrip(), self.part_header))

    def blank(self) -> None:
        self.emit("")


def _subsection_text(section: Section, index: int, total: int) -> str:
    """Deterministic clause text for subsection ``index`` (0-based) of a section."""
    number = index + 1
    ctx = {
        "worker": WORKER,
        "subject": section.subject,
        "term_a": section.term_a,
        "term_b": section.term_b,
        "prev": max(1, number - 1),
        "next": min(total, number + 1),
        "days": DAYS[index % len(DAYS)],
        "months": MONTHS[(index + len(section.title)) % len(MONTHS)],
        "cap": CAPS[(index + len(section.subject)) % len(CAPS)],
        "amount": AMOUNTS[(index + len(section.term_a)) % len(AMOUNTS)],
        "years": YEARS[index % len(YEARS)],
        "weeks": WEEKS[(index + len(section.term_b)) % len(WEEKS)],
    }
    if index == 0:
        template = PRIMARY_TEMPLATES[len(section.title) % len(PRIMARY_TEMPLATES)]
    else:
        offset = (len(section.title) * 3 + index * 5) % len(SECONDARY_TEMPLATES)
        template = SECONDARY_TEMPLATES[offset]
    return template.format(**ctx)


def _emit_section(cur: _Cursor, number: int, section: Section, flat_index: int) -> None:
    cur.blank()
    # Section line: number in column 0, at least two spaces, then a short heading.
    cur.emit(f"{number}{' ' * max(2, 6 - len(str(number)))}{section.title}")
    cur.blank()
    for index in range(section.subsections):
        body = _subsection_text(section, index, section.subsections)
        first, *rest = _wrap(f"({index + 1})  {body}", 4, 8)
        cur.emit(first, *rest)
        cur.blank()

    inserted = INSERTED_AT.get(flat_index)
    if inserted:
        _emit_inserted(cur, number, section, inserted)


def _emit_inserted(
    cur: _Cursor, host_section: int, section: Section, spec: tuple[str, str, str]
) -> None:
    """A block that inserts new provisions into another Act.

    Trap 4: the inserted provision is numbered with a letter suffix at non-zero
    indent.  Its subsections belong to *it*, not to the host section -- and
    trap 3: the quoted cross-heading that opens the block is followed by
    indented text, which is how it is told apart from a real cross-heading.
    """
    number, title, topic = spec
    ordinal = section.subsections + 1
    cur.emit(*_wrap(f"({ordinal})  In the {HOST_ACT}, after section {number[:-1]} insert—", 4, 8))
    cur.blank()
    # Quoted cross-heading: looks structural, is followed by indented text.
    cur.emit(_centre_quoted(f"{topic.capitalize()}: further provision"))
    cur.blank()
    cur.emit(f'              “{number}  {title}')
    cur.blank()
    for index, clause in enumerate(
        (
            f"An employer must secure that every {WORKER} to whom this section applies is offered {topic} in accordance with this Chapter.",
            f"The duty under subsection (1) applies in relation to each relevant reference period beginning after the coming into force of this section.",
            f"Regulations may make provision about the manner in which {topic} is to be offered, including provision about the form of an offer and the period for acceptance.",
            f"A {WORKER} may present a complaint to an employment tribunal that an employer has failed to comply with subsection (1).",
        )
    ):
        cur.emit(*_wrap(f"({index + 1})  {clause}", 14, 18))
        cur.blank()
    cur.emit(_centre_quoted("Supplementary"))
    cur.blank()
    sibling = f"{number[:-1]}{chr(ord(number[-1]) + 1)}"
    cur.emit(f'              “{sibling}  {title}: supplementary')
    cur.blank()
    cur.emit(
        *_wrap(
            f"(1)  In section {number}, references to {topic} include references to an offer made by an agent of the employer.”",
            14,
            18,
        )
    )
    cur.blank()


def _centre_quoted(text: str) -> str:
    """Indented, title-case line that mimics a cross-heading (trap 3)."""
    return " " * 14 + text


def _emit_schedule(cur: _Cursor, number: int, title: str, section_ref: int) -> None:
    cur.blank()
    cur.emit(_centre(f"SCHEDULE {number}"))
    cur.blank()
    cur.emit(_centre(f"Section {section_ref}"))
    cur.blank()
    cur.emit(_centre(title.upper()))
    cur.blank()
    for part_number, (part_title, entries) in enumerate(
        (
            ("PRELIMINARY", ("Interpretation of this Schedule", "Application of this Schedule")),
            ("PROCEDURE", ("Applications and notices", "Determinations and appeals")),
        ),
        start=1,
    ):
        cur.emit(_centre(f"PART {part_number}"))
        cur.blank()
        cur.emit(_centre(part_title))
        cur.blank()
        for index, entry in enumerate(entries, start=1):
            cur.emit(f"{index}{' ' * 5}{entry}")
            cur.blank()
            for sub in range(1, 4):
                cur.emit(
                    *_wrap(
                        f"({sub})  Paragraph {index} of this Schedule applies for the purposes of "
                        f"{title.lower()}, and regulations may make further provision about "
                        f"the matters mentioned in sub-paragraph ({max(1, sub - 1)}).",
                        4,
                        8,
                    )
                )
                cur.blank()


def _front_matter(section_index: list[tuple[int, str, str]]) -> list[str]:
    """Cover page and a table of contents that looks exactly like the body (trap 1)."""
    lines: list[str] = [""] * 6
    lines.append(_centre(SHORT_TITLE))
    lines += ["", "", _centre(CHAPTER), "", ""]
    lines.append(_centre("CONTENTS"))
    lines.append("")
    current_part = ""
    for number, part_title, section_title in section_index:
        if part_title != current_part:
            current_part = part_title
            lines += ["", _centre(f"PART {part_title.split('|')[0]}"), ""]
            lines.append(_centre(part_title.split("|")[1]))
            lines.append("")
        # Same visual shape as a body section line: this is the trap.
        lines.append(f"{number}{' ' * max(2, 6 - len(str(number)))}{section_title}")
    lines += ["", "", _centre("SCHEDULES"), ""]
    lines.append("1     Guaranteed hours: procedure")
    lines.append("2     Fair Work Agency: enforcement notices")
    lines += ["", "", ""]
    return lines


def _enacting_text() -> list[str]:
    lines = ["", "", _centre(SHORT_TITLE), "", _centre(CHAPTER), ""]
    lines += _wrap(
        "An Act to make provision about employment rights, about pay, deductions and "
        "working time, about protection from dismissal and detriment, about trade "
        "unions and industrial relations, and about the enforcement of labour market "
        "legislation; and for connected purposes.",
        0,
        0,
    )
    lines += ["", f"{' ' * max(0, PAGE_WIDTH - len(ROYAL_ASSENT) - 6)}[{ROYAL_ASSENT}]", ""]
    lines += _wrap(
        "BE IT ENACTED by the King's most Excellent Majesty, by and with the advice "
        "and consent of the Lords Spiritual and Temporal, and Commons, in this "
        "present Parliament assembled, and by the authority of the same, as follows:—",
        0,
        0,
    )
    lines.append("")
    return lines


def render() -> str:
    """Build the whole document as ``pdftotext -layout`` would render it."""
    cur = _Cursor(lines=[])
    section_index: list[tuple[int, str, str]] = []

    number = 0
    flat_index = 0
    for part in PARTS:
        cur.part_header = f"Part {part.number} — {part.title.capitalize()}"
        cur.blank()
        cur.emit(_centre(f"PART {part.number}"))
        cur.blank()
        cur.emit(_centre(part.title))
        cur.blank()
        for theme in part.themes:
            # A real cross-heading: centred, title case, immediately followed by
            # a column-0 section line.
            cur.blank()
            cur.emit(_centre(theme.heading))
            for section in theme.sections:
                number += 1
                section_index.append((number, f"{part.number}|{part.title}", section.title))
                _emit_section(cur, number, section, flat_index)
                flat_index += 1

    cur.part_header = "Schedules"
    _emit_schedule(cur, 1, "Guaranteed hours: procedure", 1)
    _emit_schedule(cur, 2, "Fair Work Agency: enforcement notices", 121)

    # ---- paginate ---------------------------------------------------------
    pages: list[list[str]] = []
    front = _front_matter(section_index) + _enacting_text()
    for start in range(0, len(front), PAGE_LINES):
        pages.append(front[start : start + PAGE_LINES])

    body = cur.lines
    page_number = len(pages)
    for start in range(0, len(body), PAGE_LINES - 4):
        chunk = body[start : start + PAGE_LINES - 4]
        page_number += 1
        header = chunk[0][1] if chunk else ""
        # Running headers, in mixed case.  A case-insensitive filter for these
        # would also eat the real ``PART n`` markers (trap 2).
        page = [
            f"{page_number}{' ' * 6}{SHORT_TITLE} (c. 14)",
            f"{' ' * 7}{header}",
            "",
        ]
        page += [line for line, _ in chunk]
        pages.append(page)

    return "\f".join("\n".join(page).rstrip() + "\n" for page in pages)


@operator_error_exit
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/corpus.layout.txt"),
        help="destination file (default: data/corpus.layout.txt)",
    )
    args = parser.parse_args(argv)
    text = render()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    pages = text.count("\f") + 1
    print(f"wrote {args.out} — {len(text):,} chars, {pages} pages, {text.count(chr(10)):,} lines")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
