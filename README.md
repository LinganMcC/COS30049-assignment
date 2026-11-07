## COS30049 Assignment 3

# Threat Model
- Who would want to attack your application?
Users trying to use AI generated text to pass of as human
Users trying to discredit the tool by showing cases where it fails
Users curious of the limits of the model
Attackers who just target any public website either for denial of service, compute abuse or weakness probing

- What are they trying to achieve?
To get a false reading for their input text to:
    Evade detection
    Slander someone or the tool
Learn how the model works
Render the tool unusable with high-volume requests
Compromise the site with injections

- What can they actually do to it?
A user can submit their text multiple times with different variations until it passes as human
Flood the endpoint with requests by inputting massive text
Command injection
Model file can replaced with a malicious file or a model that would incorrectly classify input text

