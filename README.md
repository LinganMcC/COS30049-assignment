# COS30049 Assignment 3

## Threat Model
Who would want to attack your application?
- Users trying to use AI generated text to pass of as human
- Users trying to discredit the tool by showing cases where it fails
- Users curious of the limits of the model
- Attackers who just target any public website either for denial of service, compute abuse or weakness probing

What are they trying to achieve?
- To get a false reading for their input text to:
    - Evade detection
    - Slander someone or the tool
- Learn how the model works
- Render the tool unusable with high-volume requests
- Compromise the site with injections

What can they actually do to it?
- A user can submit their text multiple times with different variations until it passes as human
- Flood the endpoint with requests by inputting massive text
- XSS Script injection that replaces the models output with its own output
- Model file can be replaced with a malicious file or a model that could alter the output result or deploy harmful script

## Security Defences (7 Points Total)
### Misconfiguration and Vulnerable Dependencies (Intermediate/3 Points)
Reasoning
- Misconfigurations such as missing security headers leave the risk of possible XSS attacks
- Including no input length cap or rate limit opens the risk of request flooding or model probing
- Model tampering can be avoided with proper configurations such as disabling debug mode and using libraries that are up-to-date with security patches

Cost
- Limiting input length and rate limiting can hinder legitimate users trying to use the website. However, The length limit should cover most standard essay lengths and many other models limit rate as well and it is necessary to prevent model probing and request flooding

### Denial of Service (Advanced/4 Points)
Reasoning
- Prevents resource exhaustion caused by request flooding and model probing with large input text

Cost
- Legitimate users who are trying to use the tool may meet the 503 error due to a large burst of requests, this is unavoidable to defend against 

### Risks not Included
SQL Injection
- We are not using an SQL databases

Broken Access Control
- We are not implementing different user accounts

Unsafe File Upload
- We are only allowing pasted/typed text, no file upload option