# Error analysis - HC3

Feature contribution values below are exact contributions to a selected sentence's LR linear logit.
They are not presented as an exact decomposition of the pooled document probability.

## False positives: Human predicted AI
Total FP: 259

### `hc3::c887c0e777ac7ad41423b68f949a8c72::human::0` — document score 0.959
Selected sentence probability: 0.959
Selected sentence: Airbus SAS (, , , ) is an aircraft manufacturing subsidiary of EADS , a European aerospace company.

Strongest sentence-logit pushes toward AI: comma_rate=+2.235, long_word_ratio=+1.131, rel_len_deviation=+0.320, unique_word_ratio=+0.194
Strongest sentence-logit pushes toward Human: noun_ratio=-0.334, conjunction_ratio=-0.163, starts_with_transition=-0.081, pronoun_ratio=-0.071

**Your notes:** _What pattern do you see?_

### `hc3::2a0d4e1c9eaee4b2f61dfcc91ca633c5::human::0` — document score 0.917
Selected sentence probability: 0.966
Selected sentence: South Africa has eleven official languages : Afrikaans , English , Ndebele , Northern Sotho , Sotho , Swazi , Tswana , Tsonga , Venda , Xhosa and Zulu.

Strongest sentence-logit pushes toward AI: comma_rate=+3.640, long_word_ratio=+0.461, auxiliary_ratio=+0.339, adverb_ratio=+0.151
Strongest sentence-logit pushes toward Human: noun_ratio=-0.607, func_word_ratio=-0.172, starts_with_transition=-0.081, adjective_ratio=-0.077

**Your notes:** _What pattern do you see?_

### `hc3::a46b8063368de991286aafac76dcae5c::human::0` — document score 0.897
Selected sentence probability: 0.897
Selected sentence: Shoulder impingement syndrome, also called painful arc syndrome, supraspinatus syndrome, swimmer's shoulder, and thrower's shoulder, is a clinical syndrome which occurs when the tendons of the rotator cuff muscles become irritated and inflamed as they pass through the subacromial space, the passage beneath the acromion .

Strongest sentence-logit pushes toward AI: long_word_ratio=+1.358, comma_rate=+0.758, rel_len_deviation=+0.320, auxiliary_ratio=+0.248
Strongest sentence-logit pushes toward Human: unique_word_ratio=-0.389, noun_ratio=-0.183, starts_with_transition=-0.081, func_word_ratio=-0.049

**Your notes:** _What pattern do you see?_

### `hc3::fe4f2057a17feef4429a80ef46a05069::human::0` — document score 0.869
Selected sentence probability: 0.869
Selected sentence: The black-and-white series originally ran from October 15, 1951, to May 6, 1957, on the Columbia Broadcasting System (CBS).

Strongest sentence-logit pushes toward AI: comma_rate=+1.621, auxiliary_ratio=+0.339, rel_len_deviation=+0.320, adjective_ratio=+0.106
Strongest sentence-logit pushes toward Human: noun_ratio=-0.196, starts_with_transition=-0.081, pronoun_ratio=-0.071, apostrophe_rate=-0.070

**Your notes:** _What pattern do you see?_

### `hc3::ec16071ae1a2cf8a609e12f76b899020::human::0` — document score 0.829
Selected sentence probability: 0.829
Selected sentence: Suicide is a fictional character from TNA Impact!

Strongest sentence-logit pushes toward AI: exclaim=+1.452, long_word_ratio=+0.733, rel_len_deviation=+0.320, unique_word_ratio=+0.194
Strongest sentence-logit pushes toward Human: comma_rate=-0.371, noun_ratio=-0.292, auxiliary_ratio=-0.171, conjunction_ratio=-0.163

**Your notes:** _What pattern do you see?_

## False negatives: AI predicted Human
Total FN: 530

### `hc3::1a289eedbdb378b15611951c96d028c2::chatgpt::0` — document score 0.199
Selected sentence probability: 0.101
Selected sentence: It's kind of like when you open a can of soda and the gas starts to come out of the can as soon as you open it - the same thing is happening when you add ice to your glass of soda, except that the cold temperature of the ice is causing the gases to come out of solution even faster.

Strongest sentence-logit pushes toward AI: auxiliary_ratio=+0.203, conjunction_ratio=+0.068, func_word_ratio=+0.057, noun_ratio=+0.034
Strongest sentence-logit pushes toward Human: unique_word_ratio=-0.812, long_word_ratio=-0.685, rel_len_deviation=-0.363, comma_rate=-0.230

**Your notes:** _What pattern do you see?_

### `hc3::59a0743c862b0d4b05cd07b99ab6da20::chatgpt::0` — document score 0.201
Selected sentence probability: 0.101
Selected sentence: In other words, it is the value of a future payment or series of payments in today's dollars, adjusted for the time value of money and any expected changes in the value of money (inflation).To calculate the present value of a single cash flow, you can use the following formula:PV = CF / (1 + r)^tWhere:PV is the present valueCF is the future cash flowr is the discount rate or required rate of returnt is the number of periods until the cash flow is receivedFor example, let's say you expect to receive $100 in one year, and your required rate of return is 10%.

Strongest sentence-logit pushes toward AI: adverb_ratio=+0.151, comma_rate=+0.044, func_word_ratio=+0.035, n_words=+0.034
Strongest sentence-logit pushes toward Human: rel_len_deviation=-1.087, unique_word_ratio=-0.912, noun_ratio=-0.137, starts_with_transition=-0.081

**Your notes:** _What pattern do you see?_

### `hc3::499d47feb954fb3594272b16bee70d58::chatgpt::0` — document score 0.221
Selected sentence probability: 0.094
Selected sentence: Here are a few more benefits of nuclear energy:\n\nIt is a reliable source of energy: Nuclear power plants can operate around the clock, providing a steady and reliable source of electricity.\n\nIt is efficient: Nuclear power plants can generate a lot of electricity using very small amounts of fuel.\n\nIt is cost-effective: Nuclear energy can be a cost-effective way to produce electricity, especially when compared to other sources of energy like coal or natural gas.\n\nIt is safe: Nuclear power plants are designed to be safe and secure, with multiple layers of protection in place to prevent accidents.\n\nOverall, nuclear energy is a good choice for producing electricity because it is clean, efficient, reliable, and cost-effective.

Strongest sentence-logit pushes toward AI: adjective_ratio=+0.440, long_word_ratio=+0.333, comma_rate=+0.103, adverb_ratio=+0.079
Strongest sentence-logit pushes toward Human: rel_len_deviation=-1.656, unique_word_ratio=-1.002, auxiliary_ratio=-0.144, starts_with_transition=-0.081

**Your notes:** _What pattern do you see?_

### `hc3::5fb62ca47ca7fd2817a221d880f1a32a::chatgpt::2` — document score 0.222
Selected sentence probability: 0.156
Selected sentence: This means that a dwarf planet has not gravitationally eliminated other objects from its orbit, and may share its orbit with other celestial bodies.\n\nTo be classified as a dwarf planet, a celestial body must meet the following criteria:\n\nIt must orbit the Sun.\nIt must be large enough to be round due to its own gravity.\nIt must not be a satellite (moon) of another planet.\nIt must not have "cleared its neighborhood" of other celestial bodies.\nThere are currently five recognized dwarf planets in our Solar System: Ceres, Pluto, Eris, Makemake, and Haumea.

Strongest sentence-logit pushes toward AI: comma_rate=+0.153, adjective_ratio=+0.114, adverb_ratio=+0.056, n_words=+0.032
Strongest sentence-logit pushes toward Human: unique_word_ratio=-0.726, rel_len_deviation=-0.423, auxiliary_ratio=-0.300, long_word_ratio=-0.138

**Your notes:** _What pattern do you see?_

### `hc3::35486f3e1d1b7018c81898d8f07ae2ae::chatgpt::0` — document score 0.231
Selected sentence probability: 0.092
Selected sentence: This can be very tiring for new parents, but there are a few things that can help them get through it:\n\nTaking turns: If both parents are present, it can be helpful for them to take turns caring for the baby at night so that one person can get some rest while the other one takes care of the baby.\n\nAsking for help: It's okay to ask for help from friends, family, or even hiring a babysitter to give the parents a break.\n\nNapping during the day: While a 30-minute nap may not be enough to fully catch up on sleep, taking a short nap during the day can help new parents feel more rested and better able to handle the demands of caring for a newborn.\n\nStaying healthy: Eating well, exercising, and getting some fresh air can help new parents feel more energetic and better able to cope with sleep deprivation.\n\nSeeking support: It can be helpful for new parents to talk to each other, a therapist, or other parents who have gone through the same experience.

Strongest sentence-logit pushes toward AI: adjective_ratio=+0.127, n_words=+0.068, comma_rate=+0.045, noun_ratio=+0.045
Strongest sentence-logit pushes toward Human: rel_len_deviation=-1.061, unique_word_ratio=-0.953, long_word_ratio=-0.263, starts_with_transition=-0.081

**Your notes:** _What pattern do you see?_
