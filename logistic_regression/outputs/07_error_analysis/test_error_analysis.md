# Error analysis - TEST

Feature contribution values below are exact contributions to a selected sentence's LR linear logit.
They are not presented as an exact decomposition of the pooled document probability.

## False positives: Human predicted AI
Total FP: 247

### `drcat::2ba499b214728da6` — document score 0.682
Selected sentence probability: 0.983
Selected sentence: Motheri, fatheri, aunti, unclei, neighbori, and grandparenti depend on their cari daily.

Strongest sentence-logit pushes toward AI: comma_rate=+3.158, long_word_ratio=+0.541, auxiliary_ratio=+0.339, adjective_ratio=+0.244
Strongest sentence-logit pushes toward Human: noun_ratio=-0.292, starts_with_transition=-0.081, apostrophe_rate=-0.070, func_word_ratio=-0.058

**Your notes:** _What pattern do you see?_

### `drcat::b6289efbc7189644` — document score 0.682
Selected sentence probability: 0.951
Selected sentence: Join me, and all your fellow friends, family, and adventerers, on a once in a lifetime expierence for yourself!

Strongest sentence-logit pushes toward AI: exclaim=+1.452, comma_rate=+1.412, auxiliary_ratio=+0.339, long_word_ratio=+0.219
Strongest sentence-logit pushes toward Human: rel_len_deviation=-0.189, unique_word_ratio=-0.082, starts_with_transition=-0.081, adjective_ratio=-0.077

**Your notes:** _What pattern do you see?_

### `drcat::00573383acc7c214` — document score 0.646
Selected sentence probability: 0.890
Selected sentence: In conclusion, if you join, you help people, be with some animals, play some games, see Beautiful sights onboard, and you'd be helping the military.

Strongest sentence-logit pushes toward AI: comma_rate=+1.661, starts_with_transition=+0.921, apostrophe_rate=+0.117, long_word_ratio=+0.112
Strongest sentence-logit pushes toward Human: rel_len_deviation=-0.423, unique_word_ratio=-0.121, adjective_ratio=-0.113, exclaim=-0.028

**Your notes:** _What pattern do you see?_

### `drcat::b6f6851193daec07` — document score 0.645
Selected sentence probability: 0.963
Selected sentence: Mothers, fathers, aunts, uncles, neighbors, and grandparents depend on their cars daily.

Strongest sentence-logit pushes toward AI: comma_rate=+3.158, long_word_ratio=+0.541, auxiliary_ratio=+0.339, unique_word_ratio=+0.194
Strongest sentence-logit pushes toward Human: noun_ratio=-0.383, adjective_ratio=-0.225, starts_with_transition=-0.081, apostrophe_rate=-0.070

**Your notes:** _What pattern do you see?_

### `drcat::83ef048a38b22d45` — document score 0.644
Selected sentence probability: 0.981
Selected sentence: It's an amazing expierience!

Strongest sentence-logit pushes toward AI: exclaim=+1.452, long_word_ratio=+1.307, apostrophe_rate=+1.098, adjective_ratio=+0.479
Strongest sentence-logit pushes toward Human: comma_rate=-0.371, rel_len_deviation=-0.347, conjunction_ratio=-0.163, starts_with_transition=-0.081

**Your notes:** _What pattern do you see?_

## False negatives: AI predicted Human
Total FN: 491

### `drcat::7058655552d74c94` — document score 0.219
Selected sentence probability: 0.118
Selected sentence: The mаin reаsоn why the government should tаke immediаte аction is becаuse it will help prevent fаr-reаching cоnsequences оf this glоbаl problem.

Strongest sentence-logit pushes toward AI: adverb_ratio=+0.151, n_words=+0.006, exclaim=-0.028, unique_word_ratio=-0.038
Strongest sentence-logit pushes toward Human: long_word_ratio=-0.450, comma_rate=-0.371, adjective_ratio=-0.225, rel_len_deviation=-0.216

**Your notes:** _What pattern do you see?_

### `drcat::d23c428979635a8f` — document score 0.300
Selected sentence probability: 0.092
Selected sentence: My dad says its like when you have a leak in your roof, you gotta fix it right away or itll cost a lot more later.

Strongest sentence-logit pushes toward AI: auxiliary_ratio=+0.193, noun_ratio=+0.096, pronoun_ratio=+0.086, func_word_ratio=+0.019
Strongest sentence-logit pushes toward Human: long_word_ratio=-0.991, rel_len_deviation=-0.895, adverb_ratio=-0.228, adjective_ratio=-0.225

**Your notes:** _What pattern do you see?_

### `drcat::0fe4cf0ffd729f7c` — document score 0.305
Selected sentence probability: 0.104
Selected sentence: One place I would love to visit in this world is Puerto Rico.

Strongest sentence-logit pushes toward AI: unique_word_ratio=+0.194, adverb_ratio=+0.151, func_word_ratio=+0.019, n_words=-0.003
Strongest sentence-logit pushes toward Human: long_word_ratio=-0.991, comma_rate=-0.371, auxiliary_ratio=-0.289, adjective_ratio=-0.225

**Your notes:** _What pattern do you see?_

### `drcat::910b66be3188229a` — document score 0.315
Selected sentence probability: 0.080
Selected sentence: We can do it too.

Strongest sentence-logit pushes toward AI: noun_ratio=+0.252, unique_word_ratio=+0.194, pronoun_ratio=+0.180, func_word_ratio=+0.028
Strongest sentence-logit pushes toward Human: long_word_ratio=-0.991, auxiliary_ratio=-0.477, adverb_ratio=-0.379, comma_rate=-0.371

**Your notes:** _What pattern do you see?_

### `drcat::11e64451c78ba61c` — document score 0.317
Selected sentence probability: 0.100
Selected sentence: Note: Please keep in mind that this is a grade 7 student, so the essay should be evaluated based on the writing skills and understanding of a 12-year-old student.

Strongest sentence-logit pushes toward AI: adverb_ratio=+0.060, func_word_ratio=+0.056, n_words=+0.004, conjunction_ratio=-0.004
Strongest sentence-logit pushes toward Human: rel_len_deviation=-1.284, long_word_ratio=-0.199, adjective_ratio=-0.128, noun_ratio=-0.123

**Your notes:** _What pattern do you see?_
