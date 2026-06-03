Below is a comprehensive **typed event-extraction schema** for the 8 corporate event types. It is designed as a **discriminated union**: every extracted record has common metadata plus one event-specific `details` object.

Assumptions:

- Use `null` when a field is not stated or cannot be inferred from the article.
- Use `[]` when a repeatable field is applicable but no items are stated.
- Each record should be supportable from a **single article**.
- Multiple people or companies in the same article may generate multiple event records where the ontology defines individual-level events, e.g. CEO/CFO changes.

```ts
// ---------- Core enums ----------

type EventType =
  | "M&A_announce"
  | "M&A_complete"
  | "M&A_cancel"
  | "M&A_rumor"
  | "CEO_change"
  | "CFO_change"
  | "IPO"
  | "Stock_split";

type DatePrecision =
  | "day"
  | "month"
  | "quarter"
  | "year"
  | "approximate"
  | "unknown";

type CurrencyCode = string; // ISO 4217 code when known, e.g. USD

type Confidence = number; // 0.0 to 1.0 extractor confidence


// ---------- Shared primitive types ----------

interface DateValue {
  value: string | null; // ISO date/datetime string if known
  precision: DatePrecision; // granularity of date value
  text: string | null; // original date phrase from article
}

interface MoneyAmount {
  amount: number | null; // numeric amount if extractable
  currency: CurrencyCode | null; // currency of amount
  scale: "units" | "thousand" | "million" | "billion" | "trillion" | null; // stated scale
  text: string | null; // original money phrase
}

interface PercentageValue {
  value: number | null; // percentage as numeric value, e.g. 12.5
  text: string | null; // original percentage phrase
}

interface NumericValue {
  value: number | null; // numeric value if extractable
  unit: string | null; // unit such as shares or dollars/share
  text: string | null; // original numeric phrase
}

interface EvidenceSpan {
  field_paths: string[]; // schema fields supported by this evidence
  quote: string; // article text supporting extraction
  start_char: number | null; // character start offset if available
  end_char: number | null; // character end offset if available
}

interface ArticleSource {
  article_id: string; // corpus-specific article identifier
  url: string | null; // article URL if available
  title: string | null; // article headline
  publisher: string | null; // news outlet or wire source
  byline: string | null; // article author if stated
  publication_datetime: string | null; // ISO publication datetime
  language: string | null; // article language code
  dateline: string | null; // dateline location if stated
}

interface ExtractionMetadata {
  extractor_name: string | null; // system or model that produced record
  extraction_datetime: string | null; // ISO extraction timestamp
  schema_version: string; // schema version used
  record_confidence: Confidence | null; // overall extraction confidence
  needs_human_review: boolean; // true if extraction is ambiguous
}

interface CompanyRef {
  name: string; // company or business name as stated
  normalized_name: string | null; // canonical company name if resolved
  ticker: string | null; // stock ticker if stated or resolved
  exchange: string | null; // listing exchange if stated
  cik: string | null; // SEC CIK if available
  lei: string | null; // legal entity identifier if available
  country: string | null; // headquarters or incorporation country
  state_or_region: string | null; // headquarters state or region
  is_public: boolean | null; // whether company is public
  industry: string | null; // industry or sector if stated
  parent_company: CompanyRef | null; // parent entity if relevant
}

interface PersonRef {
  full_name: string; // person name as stated
  normalized_name: string | null; // canonical person name if resolved
  age: number | null; // age if stated
  gender: string | null; // gender if stated or explicitly known
  nationality: string | null; // nationality if stated
}

interface SecurityRef {
  security_name: string | null; // security name if stated
  security_type: string | null; // common stock, ADS, Class A, etc.
  ticker: string | null; // security ticker
  exchange: string | null; // listing exchange
  cusip: string | null; // CUSIP if available
  isin: string | null; // ISIN if available
}

interface AdvisorRef {
  advisor: CompanyRef; // advisory firm
  role: string | null; // legal, financial, placement, underwriter, etc.
  advised_party: CompanyRef | null; // party represented by advisor
}

interface RegulatoryApproval {
  authority: string | null; // regulator or agency name
  jurisdiction: string | null; // relevant country or region
  status: "required" | "pending" | "received" | "denied" | "waived" | "unknown"; // approval status
  decision_date: DateValue | null; // date of regulatory decision
  remedies_required: string[]; // required divestitures or conditions
}

interface ApprovalRequirement {
  approval_type: string; // shareholder, antitrust, court, exchange, etc.
  required: boolean | null; // whether approval is required
  status: "pending" | "received" | "failed" | "waived" | "unknown"; // approval outcome
  vote_or_decision_date: DateValue | null; // vote or decision date
  approving_party: string | null; // party or body granting approval
}

interface FinancingDetail {
  financing_required: boolean | null; // whether financing is required
  financing_committed: boolean | null; // whether financing is committed
  debt_amount: MoneyAmount | null; // debt financing amount
  equity_amount: MoneyAmount | null; // equity financing amount
  lenders_or_investors: CompanyRef[]; // financing providers
  financing_description: string | null; // textual financing summary
}


// ---------- Root event record ----------

interface CorporateEventRecord {
  event_id: string; // unique extracted-event identifier
  event_type: EventType; // event category
  source_article: ArticleSource; // article from which record is extracted
  extraction: ExtractionMetadata; // extraction provenance metadata
  event_date: DateValue; // canonical anchor date for event type
  event_date_basis: string; // why this date anchors the event
  article_reported_date: DateValue | null; // event date as reported by article
  headline_summary: string | null; // short event summary
  primary_companies: CompanyRef[]; // main companies involved
  primary_persons: PersonRef[]; // main people involved
  geography: string[]; // countries or regions materially involved
  event_status: string | null; // status such as rumored, announced, closed, effective
  is_correction_or_update: boolean | null; // whether article updates prior news
  related_prior_event_hint: string | null; // textual hint linking to prior event
  duplicate_cluster_key_hint: string | null; // optional key for cross-article deduping
  evidence: EvidenceSpan[]; // supporting article excerpts
  details:
    | MADetail
    | ExecutiveChangeDetail
    | IPODetail
    | StockSplitDetail; // event-specific payload
}


// ---------- M&A events ----------

type MAEventSubtype =
  | "M&A_announce"
  | "M&A_complete"
  | "M&A_cancel"
  | "M&A_rumor";

type MATransactionForm =
  | "merger"
  | "acquisition"
  | "asset_purchase"
  | "tender_offer"
  | "stock_for_stock_merger"
  | "spac_business_combination"
  | "minority_stake"
  | "going_private"
  | "management_buyout"
  | "other"
  | "unknown";

type MADealStatus =
  | "rumored"
  | "in_talks"
  | "proposed"
  | "announced"
  | "pending"
  | "completed"
  | "cancelled"
  | "withdrawn"
  | "blocked"
  | "rejected"
  | "unknown";

type ConsiderationType =
  | "cash"
  | "stock"
  | "cash_and_stock"
  | "debt_assumption"
  | "asset_swap"
  | "undisclosed"
  | "other"
  | "unknown";

interface TenderOfferDetail {
  is_tender_offer: boolean; // whether deal uses tender offer
  offer_price_per_share: MoneyAmount | null; // tender offer price
  minimum_acceptance_percent: PercentageValue | null; // minimum tender condition
  expiration_date: DateValue | null; // offer expiration date
  shares_sought: NumericValue | null; // number of shares sought
  tender_result_percent: PercentageValue | null; // percentage tendered or accepted
}

interface MACompetingBid {
  bidder: CompanyRef | null; // competing bidder
  bid_value: MoneyAmount | null; // competing bid value
  bid_date: DateValue | null; // date of competing bid
  bid_status: string | null; // status of competing bid
  description: string | null; // textual bid description
}

interface MATerminationDetail {
  termination_date: DateValue | null; // public termination date
  termination_effective_date: DateValue | null; // legal termination date
  termination_reason: string | null; // reason for cancellation
  termination_initiator: CompanyRef | null; // party initiating termination
  mutual_termination: boolean | null; // whether termination was mutual
  break_fee: MoneyAmount | null; // termination fee amount
  break_fee_payer: CompanyRef | null; // party paying break fee
  regulatory_blocked: boolean | null; // true if blocked by regulator
  shareholder_vote_failed: boolean | null; // true if shareholder approval failed
  mac_invoked: boolean | null; // true if MAC clause invoked
}

interface MARumorDetail {
  rumor_language: string[]; // phrases indicating rumor or uncertainty
  rumor_source_type: "named_sources" | "unnamed_sources" | "media_report" | "issuer_denial" | "activist_proposal" | "unknown"; // source style
  original_reporting_outlet: string | null; // outlet credited with rumor
  issuer_response: "no_comment" | "denial" | "confirmation" | "clarification" | "none_reported" | "unknown"; // company response
  rumored_acquirer_named: boolean | null; // whether acquirer is named
  unsolicited_proposal: boolean | null; // whether proposal is unsolicited
  activist_involved: boolean | null; // whether activist investor involved
}

interface MADetail {
  ma_event_subtype: MAEventSubtype; // specific M&A event type
  deal_name: string | null; // common transaction name
  transaction_form: MATransactionForm; // legal or economic form
  deal_status: MADealStatus; // current deal status
  acquirers: CompanyRef[]; // buyers, bidders, or SPACs
  targets: CompanyRef[]; // target companies or assets
  sellers: CompanyRef[]; // sellers or divesting parties
  divested_business_or_assets: string | null; // business unit or assets acquired
  is_cross_border: boolean | null; // whether parties span countries
  is_related_party_transaction: boolean | null; // whether parties are related
  is_spac_business_combination: boolean | null; // whether deal is SPAC combination
  controlling_interest_transfers: boolean | null; // whether control changes hands
  stake_acquired_percent: PercentageValue | null; // percentage stake acquired
  pre_transaction_stake_percent: PercentageValue | null; // buyer stake before deal
  post_transaction_stake_percent: PercentageValue | null; // buyer stake after deal
  deal_value: MoneyAmount | null; // headline transaction value
  enterprise_value: MoneyAmount | null; // enterprise value if stated
  equity_value: MoneyAmount | null; // equity value if stated
  debt_assumed: MoneyAmount | null; // debt assumed in transaction
  consideration_type: ConsiderationType; // payment form
  cash_component: MoneyAmount | null; // cash consideration amount
  stock_component_value: MoneyAmount | null; // stock consideration value
  per_share_price: MoneyAmount | null; // price per target share
  exchange_ratio: NumericValue | null; // acquirer shares per target share
  premium_percent: PercentageValue | null; // premium to market price
  premium_reference_date: DateValue | null; // date for premium comparison
  announcement_date: DateValue | null; // public announcement date
  signing_date: DateValue | null; // agreement signing date
  expected_closing_date: DateValue | null; // projected close date
  closing_date: DateValue | null; // actual completion date
  termination: MATerminationDetail | null; // cancellation-specific details
  rumor: MARumorDetail | null; // rumor-specific details
  approvals: ApprovalRequirement[]; // required deal approvals
  regulatory_approvals: RegulatoryApproval[]; // regulator-specific approvals
  material_conditions: string[]; // conditions precedent
  financing: FinancingDetail | null; // acquisition financing details
  tender_offer: TenderOfferDetail | null; // tender offer details
  competing_bids: MACompetingBid[]; // competing offers
  advisors: AdvisorRef[]; // financial or legal advisors
  expected_synergies: MoneyAmount | null; // announced synergies
  strategic_rationale: string | null; // stated reason for transaction
  filing_references: string[]; // SEC or other filing references
}


// ---------- CEO / CFO change events ----------

type ExecutiveRole = "CEO" | "CFO";

type ExecutiveChangeAction =
  | "appointment"
  | "departure"
  | "appointment_and_departure"
  | "interim_appointment"
  | "acting_appointment"
  | "promotion"
  | "role_elimination"
  | "co_role_structure_change"
  | "death_in_office"
  | "other"
  | "unknown";

type DepartureReason =
  | "resignation"
  | "retirement"
  | "termination"
  | "removal"
  | "death"
  | "medical_leave"
  | "personal_reasons"
  | "pursuing_other_opportunities"
  | "planned_succession"
  | "mutual_agreement"
  | "not_disclosed"
  | "other"
  | "unknown";

type AppointmentSource =
  | "internal_promotion"
  | "external_hire"
  | "founder"
  | "board_member"
  | "interim_existing_executive"
  | "unknown";

interface ExecutiveChangeDetail {
  company: CompanyRef; // company whose CEO or CFO changed
  role: ExecutiveRole; // affected executive role
  person: PersonRef; // person who is appointed or departing
  action: ExecutiveChangeAction; // nature of role change
  is_interim_or_acting: boolean | null; // whether appointment is interim
  is_permanent: boolean | null; // whether appointment is permanent
  is_co_ceo_or_co_cfo: boolean | null; // whether role is shared
  co_role_structure_changed: boolean | null; // whether co-role structure changed
  announcement_date: DateValue | null; // date change was announced
  effective_date: DateValue | null; // date change takes effect
  departure_date: DateValue | null; // date person leaves role
  appointment_date: DateValue | null; // date person assumes role
  predecessor: PersonRef | null; // prior role holder
  successor: PersonRef | null; // next role holder
  replaced_person: PersonRef | null; // person directly replaced
  replacement_person: PersonRef | null; // person directly replacing subject
  prior_title: string | null; // person’s previous title
  prior_company: CompanyRef | null; // person’s previous employer
  new_title: string | null; // person’s new title
  new_company: CompanyRef | null; // person’s new employer
  retains_other_roles: string[]; // roles retained after change
  relinquishes_other_roles: string[]; // roles relinquished besides CEO/CFO
  board_role_before: string | null; // board role before change
  board_role_after: string | null; // board role after change
  departure_reason: DepartureReason | null; // stated reason for departure
  appointment_source: AppointmentSource | null; // internal or external source
  succession_plan_described: boolean | null; // whether succession plan mentioned
  search_process_status: string | null; // status of search for replacement
  transition_period: string | null; // described transition arrangement
  compensation_summary: string | null; // compensation terms if stated
  filing_references: string[]; // 8-K or other filing references
}


// ---------- IPO events ----------

type IPOStage =
  | "confidential_filing"
  | "registration_filing"
  | "roadshow"
  | "pricing"
  | "first_trading_day"
  | "completed_offering"
  | "withdrawn_or_postponed"
  | "rumored_or_planned"
  | "other"
  | "unknown";

type IPOOfferingType =
  | "traditional_underwritten_ipo"
  | "direct_listing"
  | "dual_listing"
  | "spinout_ipo"
  | "privatization_relisting"
  | "other"
  | "unknown";

interface IPOUnderwriter {
  underwriter: CompanyRef; // bank or broker involved
  role: "lead_left" | "joint_bookrunner" | "bookrunner" | "co_manager" | "advisor" | "unknown"; // underwriting role
}

interface IPOPriceRange {
  low: MoneyAmount | null; // lower end of range
  high: MoneyAmount | null; // upper end of range
  midpoint: MoneyAmount | null; // midpoint if calculated or stated
}

interface IPODetail {
  issuer: CompanyRef; // company going public
  ipo_stage: IPOStage; // stage reported by article
  offering_type: IPOOfferingType; // IPO structure
  is_first_public_listing: boolean | null; // whether first public listing
  is_spac_ipo: boolean | null; // whether issuer is SPAC shell
  is_spinout_or_subsidiary_ipo: boolean | null; // whether issuer spun out
  parent_company: CompanyRef | null; // parent of spinout issuer
  registration_filing_date: DateValue | null; // S-1 or equivalent filing date
  confidential_filing_date: DateValue | null; // confidential filing date
  pricing_date: DateValue | null; // IPO pricing date
  first_trading_date: DateValue | null; // first trading date
  expected_trading_date: DateValue | null; // planned trading date
  withdrawal_or_postponement_date: DateValue | null; // withdrawn/postponed date
  exchange: string | null; // listing exchange
  ticker: string | null; // expected or actual ticker
  listing_country: string | null; // country of listing
  security: SecurityRef | null; // security being offered
  shares_offered: NumericValue | null; // total shares offered
  primary_shares: NumericValue | null; // new shares sold by issuer
  secondary_shares: NumericValue | null; // existing shares sold by holders
  greenshoe_shares: NumericValue | null; // over-allotment option shares
  price_range: IPOPriceRange | null; // proposed price range
  offer_price: MoneyAmount | null; // final IPO price
  gross_proceeds: MoneyAmount | null; // gross offering proceeds
  net_proceeds: MoneyAmount | null; // net proceeds to issuer
  implied_valuation: MoneyAmount | null; // valuation implied by IPO
  market_cap_at_ipo: MoneyAmount | null; // market capitalization at IPO
  enterprise_value_at_ipo: MoneyAmount | null; // enterprise value at IPO
  first_day_open_price: MoneyAmount | null; // opening trade price
  first_day_close_price: MoneyAmount | null; // first-day closing price
  first_day_return_percent: PercentageValue | null; // first-day price change
  use_of_proceeds: string | null; // stated use of IPO proceeds
  selling_shareholders: CompanyRef[]; // holders selling shares
  major_existing_investors: CompanyRef[]; // major pre-IPO investors
  underwriters: IPOUnderwriter[]; // IPO banks
  lockup_period_days: number | null; // lockup length in days
  employee_or_customer_allocation: string | null; // directed-share allocation info
  filing_references: string[]; // S-1, F-1, 424B, prospectus refs
}


// ---------- Stock split events ----------

type StockSplitType =
  | "forward_split"
  | "reverse_split"
  | "stock_dividend"
  | "split_like_stock_dividend"
  | "other"
  | "unknown";

interface SplitRatio {
  ratio_text: string | null; // stated ratio, e.g. 2-for-1
  new_shares: number | null; // numerator, new shares received
  old_shares: number | null; // denominator, old shares exchanged
  share_count_multiplier: number | null; // new_shares / old_shares
}

interface StockSplitDetail {
  company: CompanyRef; // company conducting split
  security: SecurityRef | null; // affected security class
  split_type: StockSplitType; // forward, reverse, or stock dividend
  ratio: SplitRatio; // split exchange ratio
  stock_dividend_percent: PercentageValue | null; // stock dividend percent if used
  announcement_date: DateValue | null; // public announcement date
  board_approval_date: DateValue | null; // board approval date
  shareholder_approval_required: boolean | null; // whether holder approval needed
  shareholder_approval_date: DateValue | null; // approval vote date
  record_date: DateValue | null; // holder-of-record date
  payable_date: DateValue | null; // distribution date
  ex_date: DateValue | null; // ex-split or ex-dividend date
  effective_date: DateValue | null; // split effective date
  trading_start_date: DateValue | null; // first split-adjusted trading date
  pre_split_shares_outstanding: NumericValue | null; // shares before split
  post_split_shares_outstanding: NumericValue | null; // shares after split
  fractional_share_treatment: string | null; // cash-in-lieu or rounding method
  authorized_share_change: string | null; // authorized share adjustment
  stated_reason: string | null; // company’s rationale
  exchange_notification_reference: string | null; // exchange notice if stated
  filing_references: string[]; // 8-K, proxy, exchange filing refs
}
```

### Design trade-offs considered

1. **Single unified record vs. separate tables**  
   I used a discriminated-union record so an extractor can emit one self-contained object per article/event, while analysts can still flatten it into relational tables later.

2. **More nullable fields rather than minimal fields**  
   Many article-level reports omit deal values, effective dates, approvals, or underwriters. The schema keeps these fields nullable so it remains comprehensive without forcing hallucinated values.

3. **M&A shared detail object**  
   `M&A_announce`, `M&A_complete`, `M&A_cancel`, and `M&A_rumor` share one `MADetail` structure because analysts often query across the full deal lifecycle. Subtype-specific fields such as `termination` and `rumor` are nullable.

4. **Article-derived event dates preserved separately**  
   `event_date` is the canonical benchmark anchor, while fields like `announcement_date`, `closing_date`, and `effective_date` preserve dates mentioned in the article.

5. **Deduplication not forced**  
   `duplicate_cluster_key_hint` and `related_prior_event_hint` support later aggregation across articles, but each record remains independently derivable from a single article.

6. **Evidence included for auditability**  
   Field-level evidence spans allow downstream validation and analyst trust, especially for ambiguous rumors, executive transitions, and deal-status changes.