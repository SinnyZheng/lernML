import re
import math

# Sample dataset (List of tuples: (email text, label))
data = [
    ("Win a free iPhone now! Click this link", "spam"),
    ("Limited offer! Buy now and get 50% off", "spam"),
    ("Hello, let's meet for coffee tomorrow", "ham"),
    ("Congratulations! You won a lottery", "spam"),
    ("Are you available for a meeting tomorrow?", "ham"),
    ("Claim your free prize today!", "spam"),
    ("Reminder: Your appointment is scheduled", "ham"),
    ("Your invoice for last month is attached", "ham")
]

# Step 1: Preprocess text (Tokenization, Lowercasing, Removing Punctuation)
def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation
    return text.split()  # Tokenize (split into words)

# Step 2: Create Vocabulary and Word Frequency Counts
word_counts = {"spam": {}, "ham": {}}  # Store word frequencies
spam_count, ham_count = 0, 0  # Count of spam and ham emails
vocab = set()  # Unique words in dataset

for text, label in data:
    words = preprocess(text)
    vocab.update(words)  # Add words to vocabulary
    if label == "spam":
        spam_count += 1
        for word in words:
            word_counts["spam"][word] = word_counts["spam"].get(word, 0) + 1
    else:
        ham_count += 1
        for word in words:
            word_counts["ham"][word] = word_counts["ham"].get(word, 0) + 1

# Step 3: Calculate Prior Probabilities
total_emails = spam_count + ham_count
p_spam = spam_count / total_emails  # P(Spam)
p_ham = ham_count / total_emails    # P(Ham)

# Step 4: Compute Word Probabilities with Laplace Smoothing
def compute_word_probabilities(word_counts, label, vocab, alpha=1):
    total_words = sum(word_counts[label].values()) + alpha * len(vocab)
    word_probs = {}
    for word in vocab:
        # P(word | label) = (word count + alpha) / (total words + alpha * vocab size)
        word_probs[word] = (word_counts[label].get(word, 0) + alpha) / total_words
    return word_probs

spam_probs = compute_word_probabilities(word_counts, "spam", vocab)
ham_probs = compute_word_probabilities(word_counts, "ham", vocab)

# Step 5: Classify New Email
def classify(email):
    words = preprocess(email)
    
    # Compute Log Probabilities to Avoid Underflow
    log_prob_spam = math.log(p_spam)
    log_prob_ham = math.log(p_ham)
    
    for word in words:
        if word in vocab:  # Ignore words not in training data
            log_prob_spam += math.log(spam_probs[word])
            log_prob_ham += math.log(ham_probs[word])
    
    return "spam" if log_prob_spam > log_prob_ham else "ham"

# Step 6: Test with New Emails
test_emails = [
    "Get a free lottery ticket now!",
    "Can we reschedule our meeting?",
    "Congratulations! You have been selected to win an iPhone",
    "Reminder: Your bank statement is available"
]

for email in test_emails:
    print(f"Email: \"{email}\" -> Classified as: {classify(email)}")
