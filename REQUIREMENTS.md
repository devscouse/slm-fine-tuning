# Requirements: Email Triage Classifier

## Objective

Fine-tune a small language model (SLM) to perform **multi-class classification** on emails. Given an email's subject line and body, the model must assign exactly **one** of four triage classes. The classes are mutually exclusive — every email belongs to a single class.

The classifier is intended to sit downstream of an existing spam filter; spam and phishing detection are out of scope.

## Classes

### 1. Attention

Emails that require the recipient to take a meaningful action. Something will be missed, delayed, or lost if the email is not acted on.

**Examples:**

- An email expecting or requesting a reply
- A request to make a payment (invoice, bill, subscription renewal)
- A calendar invite or scheduling request that needs an RSVP
- An approval request (expense report, pull request review, access request)
- A deadline or due-date reminder
- A support ticket or helpdesk issue awaiting the recipient's response

### 2. Notice

Emails that contain useful or relevant information but do not require any action from the recipient. Reading is enough.

**Examples:**

- Newsletters or curated content digests
- A reply to an email the recipient sent that does not call for further action
- Account statements or billing summaries
- FYI or informational forwards from colleagues or personal contacts

### 3. Ignore

Emails that are entirely unimportant for day-to-day activity. They exist mainly as a searchable record and can be safely skipped.

**Examples:**

- Order confirmations and purchase receipts
- Shipping or delivery status updates
- Read receipts
- Marketing offers and promotional emails
- Company terms-of-service or privacy-policy updates
- Social media notifications (LinkedIn, Twitter/X, Facebook, etc.)
- Automated system notifications (CI/CD build results, cron job summaries)
- Unsubscribe or preference-update confirmations
- Cookie policy or GDPR consent updates

### 4. Security

Emails related to account security or identity verification. These typically require an immediate, short-lived response and should be surfaced separately from general "Attention" items.

**Examples:**

- Multi-factor authentication (MFA) codes or push-notification prompts
- Email address verification requests
- Security alerts (unusual login attempt, account locked, etc.)
- Password reset or change requests
- "Login from a new device" notifications
- Suspicious activity warnings
- Recovery code or backup key notifications
