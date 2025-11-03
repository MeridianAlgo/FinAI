#!/usr/bin/env python3
"""
FinAI - Local Financial AI Framework
A comprehensive, self-contained financial advisor that runs entirely locally.
Uses custom-trained models on synthetic data + advanced rule-based logic.
No external APIs, internet access, or pre-trained models required.
"""

import sys
import os
import random
import json
import pickle
import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Standard scientific computing
import numpy as np
import pandas as pd

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# NLP
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK not available. Using basic NLP fallback.")

# Financial calculations
try:
    import numpy_financial as npf
    NPF_AVAILABLE = True
except ImportError:
    NPF_AVAILABLE = False


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

class Config:
    """Global configuration for FinAI"""
    MODEL_PATH = "finai_model.pkl"
    VECTORIZER_PATH = "finai_vectorizer.pkl"
    ENCODER_PATH = "finai_encoder.pkl"
    SCALER_PATH = "finai_scaler.pkl"
    TRAINING_SAMPLES = 10000
    RANDOM_SEED = 42
    
    # Financial assumptions
    INFLATION_RATE = 0.025  # 2.5%
    STOCK_RETURN = 0.07     # 7%
    BOND_RETURN = 0.03      # 3%
    SAVINGS_RATE = 0.04     # 4%
    MONTE_CARLO_ITERATIONS = 10000

# Set random seeds for reproducibility
random.seed(Config.RANDOM_SEED)
np.random.seed(Config.RANDOM_SEED)


# ============================================================================
# NLTK INITIALIZATION
# ============================================================================

def initialize_nltk():
    """Download required NLTK data if available"""
    if NLTK_AVAILABLE:
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading NLTK data...")
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
            except:
                print("Could not download NLTK data. Using fallback.")



# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

class SyntheticDataGenerator:
    """Generate extensive synthetic financial data for training"""
    
    # Define all financial modules and their query patterns
    MODULES = {
        'budgeting': [
            'help me create a budget', 'track my expenses', 'budget planning',
            'how to manage monthly expenses', 'spending analysis', 'cash flow',
            'income allocation', 'expense categories', 'budget optimization',
            'where is my money going', 'reduce spending', 'budget breakdown'
        ],
        'savings': [
            'emergency fund', 'save money', 'savings account', 'high yield savings',
            'how much should I save', 'savings goals', 'compound interest',
            'grow my savings', 'savings calculator', 'rainy day fund',
            'short term savings', 'savings strategy'
        ],
        'debt': [
            'pay off debt', 'credit card debt', 'student loans', 'mortgage payment',
            'debt consolidation', 'refinancing', 'snowball method', 'avalanche method',
            'debt free', 'loan payoff', 'interest rates', 'debt management',
            'reduce debt', 'debt to income ratio'
        ],
        'investing': [
            'invest money', 'stock market', 'portfolio', 'diversification',
            'asset allocation', 'ETF', 'mutual funds', 'index funds', 'bonds',
            'real estate investment', 'REIT', 'investment strategy', 'risk tolerance',
            'rebalancing portfolio', 'dollar cost averaging', 'ESG investing'
        ],
        'retirement': [
            'retirement planning', '401k', 'IRA', 'Roth IRA', 'pension',
            'retire early', 'retirement savings', 'social security', 'retirement age',
            'withdrawal rate', 'retirement income', 'nest egg', 'FIRE movement',
            'retirement calculator', 'retirement goals'
        ],
        'tax': [
            'tax planning', 'tax deductions', 'tax credits', 'tax bracket',
            'capital gains tax', 'tax loss harvesting', 'HSA', '529 plan',
            'tax advantaged accounts', 'reduce taxes', 'tax optimization',
            'filing status', 'itemized deductions', 'standard deduction'
        ],
        'insurance': [
            'life insurance', 'health insurance', 'disability insurance',
            'auto insurance', 'home insurance', 'umbrella policy', 'coverage needs',
            'insurance premiums', 'term vs whole life', 'insurance planning',
            'risk management', 'insurance quotes'
        ],
        'estate': [
            'estate planning', 'will', 'trust', 'beneficiary', 'inheritance',
            'power of attorney', 'probate', 'estate taxes', 'wealth transfer',
            'legacy planning', 'living trust', 'estate distribution'
        ],
        'credit': [
            'credit score', 'improve credit', 'credit report', 'credit utilization',
            'build credit', 'credit monitoring', 'FICO score', 'credit history',
            'credit cards', 'credit repair', 'payment history'
        ],
        'realestate': [
            'buy a house', 'rent vs buy', 'mortgage', 'down payment',
            'closing costs', 'home affordability', 'property investment',
            'rental property', 'real estate ROI', 'home equity', 'refinance mortgage'
        ],
        'education': [
            'college savings', '529 plan', 'student loans', 'education funding',
            'tuition costs', 'financial aid', 'scholarship', 'college planning',
            'education expenses', 'Coverdell ESA'
        ],
        'business': [
            'start a business', 'business loan', 'cash flow management',
            'profit and loss', 'business expenses', 'startup costs', 'bookkeeping',
            'business funding', 'venture capital', 'break even analysis'
        ],
        'alternative': [
            'cryptocurrency', 'bitcoin', 'NFT', 'alternative investments',
            'peer to peer lending', 'commodities', 'gold investment',
            'crypto trading', 'digital assets', 'high risk investments'
        ],
        'goals': [
            'financial goals', 'save for wedding', 'buy a car', 'vacation fund',
            'life goals', 'milestone planning', 'goal tracking', 'SMART goals',
            'financial planning', 'major purchase'
        ],
        'economy': [
            'inflation', 'interest rates', 'economic cycle', 'recession',
            'market volatility', 'federal reserve', 'economic indicators',
            'purchasing power', 'cost of living', 'economic trends'
        ],
        'charity': [
            'charitable giving', 'donations', 'donor advised fund', 'philanthropy',
            'tax deductible donations', 'charitable planning', 'giving strategy',
            'nonprofit donations', 'charitable impact'
        ],
        'behavioral': [
            'emotional investing', 'investment psychology', 'loss aversion',
            'behavioral finance', 'financial discipline', 'investment biases',
            'market timing', 'panic selling', 'financial habits'
        ],
        'career': [
            'salary negotiation', 'side hustle', 'career change', 'income growth',
            'job offer', 'raise request', 'freelancing', 'passive income',
            'career planning', 'income optimization'
        ]
    }

    
    @staticmethod
    def generate_training_data(num_samples: int = 10000) -> Tuple[List[str], List[str]]:
        """Generate synthetic query-label pairs for training"""
        queries = []
        labels = []
        
        # Generate samples for each module
        samples_per_module = num_samples // len(SyntheticDataGenerator.MODULES)
        
        for module, patterns in SyntheticDataGenerator.MODULES.items():
            for _ in range(samples_per_module):
                # Pick a base pattern
                base = random.choice(patterns)
                
                # Add variations
                variations = [
                    base,
                    f"I need help with {base}",
                    f"Can you help me {base}?",
                    f"What about {base}",
                    f"Tell me about {base}",
                    f"How do I {base}",
                    f"Advice on {base}",
                    f"Question about {base}",
                    f"{base} please",
                    f"Looking for {base} information",
                ]
                
                query = random.choice(variations)
                
                # Add noise and complexity
                if random.random() > 0.7:
                    prefixes = ["Hi, ", "Hello, ", "Hey, ", "Quick question: ", ""]
                    query = random.choice(prefixes) + query
                
                if random.random() > 0.8:
                    suffixes = [" thanks", " please", "!", "?", " urgently", ""]
                    query = query + random.choice(suffixes)
                
                queries.append(query.lower())
                labels.append(module)
        
        # Add some multi-topic queries
        for _ in range(num_samples // 20):
            modules = random.sample(list(SyntheticDataGenerator.MODULES.keys()), 2)
            pattern1 = random.choice(SyntheticDataGenerator.MODULES[modules[0]])
            pattern2 = random.choice(SyntheticDataGenerator.MODULES[modules[1]])
            query = f"{pattern1} and {pattern2}"
            queries.append(query.lower())
            labels.append(modules[0])  # Primary module
        
        return queries, labels
    
    @staticmethod
    def generate_financial_scenarios(num_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic financial scenarios for regression tasks"""
        data = []
        
        for _ in range(num_samples):
            scenario = {
                'age': random.randint(22, 70),
                'income': random.randint(30000, 250000),
                'expenses': random.randint(20000, 150000),
                'debt': random.randint(0, 100000),
                'savings': random.randint(0, 500000),
                'risk_tolerance': random.choice([1, 2, 3, 4, 5]),  # 1=conservative, 5=aggressive
                'years_to_retirement': random.randint(5, 40),
                'dependents': random.randint(0, 4),
                'credit_score': random.randint(550, 850),
            }
            
            # Calculate derived features
            scenario['savings_rate'] = (scenario['income'] - scenario['expenses']) / scenario['income'] if scenario['income'] > 0 else 0
            scenario['debt_to_income'] = scenario['debt'] / scenario['income'] if scenario['income'] > 0 else 0
            scenario['emergency_fund_months'] = scenario['savings'] / (scenario['expenses'] / 12) if scenario['expenses'] > 0 else 0
            
            # Target: recommended stock allocation percentage
            base_stock = max(0, min(100, 110 - scenario['age']))
            risk_adjustment = (scenario['risk_tolerance'] - 3) * 10
            scenario['recommended_stock_allocation'] = max(0, min(100, base_stock + risk_adjustment))
            
            data.append(scenario)
        
        return pd.DataFrame(data)



# ============================================================================
# NLP UTILITIES
# ============================================================================

class NLPProcessor:
    """Natural language processing utilities"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer() if NLTK_AVAILABLE else None
        try:
            self.stop_words = set(stopwords.words('english')) if NLTK_AVAILABLE else set()
        except:
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'])
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text"""
        if NLTK_AVAILABLE:
            try:
                return word_tokenize(text.lower())
            except:
                pass
        return text.lower().split()
    
    def preprocess(self, text: str) -> str:
        """Preprocess text for analysis"""
        # Lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stopwords and lemmatize
        if self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words]
        else:
            tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        return ' '.join(tokens)
    
    def extract_numbers(self, text: str) -> List[float]:
        """Extract numerical values from text"""
        # Find numbers including decimals and commas
        pattern = r'\$?\d+(?:,\d{3})*(?:\.\d+)?'
        matches = re.findall(pattern, text)
        numbers = []
        for match in matches:
            try:
                # Remove $ and commas
                num = float(match.replace('$', '').replace(',', ''))
                numbers.append(num)
            except:
                pass
        return numbers
    
    def detect_sentiment(self, text: str) -> str:
        """Simple sentiment detection"""
        positive_words = ['good', 'great', 'excellent', 'happy', 'excited', 'confident', 'optimistic']
        negative_words = ['bad', 'worried', 'concerned', 'anxious', 'stressed', 'difficult', 'struggling']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return 'positive'
        elif neg_count > pos_count:
            return 'negative'
        return 'neutral'



# ============================================================================
# CUSTOM AI MODEL
# ============================================================================

class FinAIModel:
    """Custom trained model for intent classification and predictions"""
    
    def __init__(self):
        self.intent_classifier = None
        self.vectorizer = None
        self.label_encoder = None
        self.allocation_model = None
        self.scaler = None
        self.is_trained = False
    
    def train(self, num_samples: int = 10000):
        """Train the model on synthetic data"""
        print(f"\n🧠 Training FinAI model on {num_samples} synthetic samples...")
        print("This may take a minute...")
        
        # Generate training data for intent classification
        queries, labels = SyntheticDataGenerator.generate_training_data(num_samples)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Vectorize text
        self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(queries)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, encoded_labels, test_size=0.2, random_state=Config.RANDOM_SEED
        )
        
        # Train intent classifier
        self.intent_classifier = MultinomialNB(alpha=0.1)
        self.intent_classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.intent_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✓ Intent classifier trained with {accuracy*100:.1f}% accuracy")
        
        # Train allocation recommendation model
        print("Training portfolio allocation model...")
        scenario_data = SyntheticDataGenerator.generate_financial_scenarios(1000)
        
        feature_cols = ['age', 'income', 'expenses', 'debt', 'savings', 
                       'risk_tolerance', 'years_to_retirement', 'dependents', 
                       'credit_score', 'savings_rate', 'debt_to_income']
        
        X_alloc = scenario_data[feature_cols].values
        y_alloc = scenario_data['recommended_stock_allocation'].values
        
        self.scaler = StandardScaler()
        X_alloc_scaled = self.scaler.fit_transform(X_alloc)
        
        self.allocation_model = RandomForestRegressor(n_estimators=50, random_state=Config.RANDOM_SEED)
        self.allocation_model.fit(X_alloc_scaled, y_alloc)
        
        print("✓ Allocation model trained")
        
        self.is_trained = True
        print("✓ Model training complete!\n")
    
    def predict_intent(self, query: str) -> Tuple[str, float]:
        """Predict the intent/module from a query"""
        if not self.is_trained:
            return 'general', 0.5
        
        X = self.vectorizer.transform([query])
        proba = self.intent_classifier.predict_proba(X)[0]
        predicted_idx = np.argmax(proba)
        confidence = proba[predicted_idx]
        intent = self.label_encoder.inverse_transform([predicted_idx])[0]
        
        return intent, confidence
    
    def predict_allocation(self, user_profile: Dict) -> float:
        """Predict recommended stock allocation based on user profile"""
        if not self.is_trained or self.allocation_model is None:
            # Fallback to simple rule
            return max(0, min(100, 110 - user_profile.get('age', 40)))
        
        features = [
            user_profile.get('age', 40),
            user_profile.get('income', 60000),
            user_profile.get('expenses', 40000),
            user_profile.get('debt', 10000),
            user_profile.get('savings', 50000),
            user_profile.get('risk_tolerance', 3),
            user_profile.get('years_to_retirement', 25),
            user_profile.get('dependents', 0),
            user_profile.get('credit_score', 700),
            user_profile.get('savings_rate', 0.2),
            user_profile.get('debt_to_income', 0.2),
        ]
        
        X = self.scaler.transform([features])
        allocation = self.allocation_model.predict(X)[0]
        return max(0, min(100, allocation))
    
    def save(self):
        """Save trained model to disk"""
        if self.is_trained:
            with open(Config.MODEL_PATH, 'wb') as f:
                pickle.dump(self.intent_classifier, f)
            with open(Config.VECTORIZER_PATH, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            with open(Config.ENCODER_PATH, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            if self.allocation_model:
                with open(Config.SCALER_PATH, 'wb') as f:
                    pickle.dump((self.allocation_model, self.scaler), f)
    
    def load(self) -> bool:
        """Load trained model from disk"""
        try:
            with open(Config.MODEL_PATH, 'rb') as f:
                self.intent_classifier = pickle.load(f)
            with open(Config.VECTORIZER_PATH, 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(Config.ENCODER_PATH, 'rb') as f:
                self.label_encoder = pickle.load(f)
            if os.path.exists(Config.SCALER_PATH):
                with open(Config.SCALER_PATH, 'rb') as f:
                    self.allocation_model, self.scaler = pickle.load(f)
            self.is_trained = True
            return True
        except:
            return False



# ============================================================================
# FINANCIAL CALCULATORS
# ============================================================================

class FinancialCalculators:
    """Collection of financial calculation utilities"""
    
    @staticmethod
    def compound_interest(principal: float, rate: float, time: float, compounds_per_year: int = 12) -> float:
        """Calculate compound interest: FV = PV * (1 + r/n)^(n*t)"""
        return principal * (1 + rate / compounds_per_year) ** (compounds_per_year * time)
    
    @staticmethod
    def future_value_annuity(payment: float, rate: float, periods: int) -> float:
        """Calculate future value of annuity (regular payments)"""
        if rate == 0:
            return payment * periods
        return payment * (((1 + rate) ** periods - 1) / rate)
    
    @staticmethod
    def present_value_annuity(payment: float, rate: float, periods: int) -> float:
        """Calculate present value of annuity"""
        if rate == 0:
            return payment * periods
        return payment * ((1 - (1 + rate) ** -periods) / rate)
    
    @staticmethod
    def loan_payment(principal: float, annual_rate: float, years: float) -> float:
        """Calculate monthly loan payment"""
        if NPF_AVAILABLE:
            monthly_rate = annual_rate / 12
            num_payments = years * 12
            return -npf.pmt(monthly_rate, num_payments, principal)
        else:
            # Manual calculation
            monthly_rate = annual_rate / 12
            num_payments = years * 12
            if monthly_rate == 0:
                return principal / num_payments
            return principal * (monthly_rate * (1 + monthly_rate) ** num_payments) / \
                   ((1 + monthly_rate) ** num_payments - 1)
    
    @staticmethod
    def amortization_schedule(principal: float, annual_rate: float, years: float) -> pd.DataFrame:
        """Generate loan amortization schedule"""
        monthly_payment = FinancialCalculators.loan_payment(principal, annual_rate, years)
        monthly_rate = annual_rate / 12
        num_payments = int(years * 12)
        
        schedule = []
        balance = principal
        
        for month in range(1, num_payments + 1):
            interest_payment = balance * monthly_rate
            principal_payment = monthly_payment - interest_payment
            balance -= principal_payment
            
            schedule.append({
                'Month': month,
                'Payment': monthly_payment,
                'Principal': principal_payment,
                'Interest': interest_payment,
                'Balance': max(0, balance)
            })
        
        return pd.DataFrame(schedule)
    
    @staticmethod
    def monte_carlo_simulation(initial_investment: float, annual_contribution: float,
                               years: int, mean_return: float, std_dev: float,
                               iterations: int = 10000) -> Dict:
        """Run Monte Carlo simulation for investment projections"""
        results = []
        
        for _ in range(iterations):
            balance = initial_investment
            for year in range(years):
                # Random return for this year
                annual_return = np.random.normal(mean_return, std_dev)
                balance = balance * (1 + annual_return) + annual_contribution
            results.append(balance)
        
        results = np.array(results)
        
        return {
            'mean': np.mean(results),
            'median': np.median(results),
            'std': np.std(results),
            'percentile_10': np.percentile(results, 10),
            'percentile_25': np.percentile(results, 25),
            'percentile_75': np.percentile(results, 75),
            'percentile_90': np.percentile(results, 90),
            'min': np.min(results),
            'max': np.max(results),
        }
    
    @staticmethod
    def retirement_needs(current_age: int, retirement_age: int, annual_expenses: float,
                        inflation_rate: float = 0.025, withdrawal_rate: float = 0.04) -> Dict:
        """Calculate retirement needs using 4% rule"""
        years_to_retirement = retirement_age - current_age
        
        # Adjust expenses for inflation
        future_annual_expenses = annual_expenses * (1 + inflation_rate) ** years_to_retirement
        
        # Calculate needed nest egg
        nest_egg_needed = future_annual_expenses / withdrawal_rate
        
        return {
            'years_to_retirement': years_to_retirement,
            'current_annual_expenses': annual_expenses,
            'future_annual_expenses': future_annual_expenses,
            'nest_egg_needed': nest_egg_needed,
            'withdrawal_rate': withdrawal_rate
        }
    
    @staticmethod
    def tax_bracket_calculation(income: float, filing_status: str = 'single') -> Dict:
        """Calculate federal tax based on 2023 brackets (simplified)"""
        # 2023 tax brackets (simplified)
        brackets = {
            'single': [
                (11000, 0.10),
                (44725, 0.12),
                (95375, 0.22),
                (182100, 0.24),
                (231250, 0.32),
                (578125, 0.35),
                (float('inf'), 0.37)
            ],
            'married': [
                (22000, 0.10),
                (89050, 0.12),
                (190750, 0.22),
                (364200, 0.24),
                (462500, 0.32),
                (693750, 0.35),
                (float('inf'), 0.37)
            ]
        }
        
        bracket_list = brackets.get(filing_status, brackets['single'])
        
        tax = 0
        previous_limit = 0
        effective_rate = 0
        marginal_rate = 0
        
        for limit, rate in bracket_list:
            if income > previous_limit:
                taxable_in_bracket = min(income, limit) - previous_limit
                tax += taxable_in_bracket * rate
                marginal_rate = rate
                previous_limit = limit
            else:
                break
        
        effective_rate = tax / income if income > 0 else 0
        
        return {
            'gross_income': income,
            'total_tax': tax,
            'effective_rate': effective_rate,
            'marginal_rate': marginal_rate,
            'after_tax_income': income - tax
        }



# ============================================================================
# CONTEXT MANAGER
# ============================================================================

class ConversationContext:
    """Manage conversation context and user profile"""
    
    def __init__(self):
        self.history: List[Dict] = []
        self.user_profile: Dict = {
            'age': None,
            'income': None,
            'expenses': None,
            'savings': None,
            'debt': None,
            'risk_tolerance': None,
            'dependents': None,
            'credit_score': None,
            'years_to_retirement': None,
            'savings_rate': None,
            'debt_to_income': None,
        }
        self.current_topic: Optional[str] = None
        self.preferences: Dict = {}
    
    def add_message(self, role: str, content: str):
        """Add message to conversation history"""
        self.history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
    
    def update_profile(self, **kwargs):
        """Update user profile with new information"""
        for key, value in kwargs.items():
            if key in self.user_profile:
                self.user_profile[key] = value
        
        # Calculate derived values
        if self.user_profile['income'] and self.user_profile['expenses']:
            self.user_profile['savings_rate'] = (
                self.user_profile['income'] - self.user_profile['expenses']
            ) / self.user_profile['income']
        
        if self.user_profile['debt'] and self.user_profile['income']:
            self.user_profile['debt_to_income'] = (
                self.user_profile['debt'] / self.user_profile['income']
            )
    
    def get_recent_context(self, num_messages: int = 5) -> str:
        """Get recent conversation context"""
        recent = self.history[-num_messages:] if len(self.history) > num_messages else self.history
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent])
    
    def extract_info_from_query(self, query: str, nlp: NLPProcessor):
        """Extract and update profile information from query"""
        numbers = nlp.extract_numbers(query)
        query_lower = query.lower()
        
        # Try to extract age
        if 'age' in query_lower or 'years old' in query_lower or "i'm" in query_lower:
            for num in numbers:
                if 18 <= num <= 100:
                    self.update_profile(age=int(num))
                    break
        
        # Try to extract income
        if 'income' in query_lower or 'salary' in query_lower or 'earn' in query_lower or 'make' in query_lower:
            for num in numbers:
                if num > 10000:  # Likely annual income
                    self.update_profile(income=num)
                    break
        
        # Try to extract expenses
        if 'expense' in query_lower or 'spend' in query_lower or 'cost' in query_lower:
            for num in numbers:
                if num > 1000:
                    self.update_profile(expenses=num)
                    break
        
        # Try to extract savings
        if 'save' in query_lower or 'savings' in query_lower:
            for num in numbers:
                if num > 0:
                    self.update_profile(savings=num)
                    break
        
        # Try to extract debt
        if 'debt' in query_lower or 'owe' in query_lower or 'loan' in query_lower:
            for num in numbers:
                if num > 0:
                    self.update_profile(debt=num)
                    break



# ============================================================================
# RESPONSE GENERATOR
# ============================================================================

class ResponseGenerator:
    """Generate natural, contextual responses"""
    
    def __init__(self, model: FinAIModel, nlp: NLPProcessor, calc: FinancialCalculators):
        self.model = model
        self.nlp = nlp
        self.calc = calc
    
    def generate_response(self, query: str, context: ConversationContext) -> str:
        """Generate comprehensive response based on query and context"""
        
        # Predict intent
        intent, confidence = self.model.predict_intent(query)
        context.current_topic = intent
        
        # Extract information from query
        context.extract_info_from_query(query, self.nlp)
        
        # Detect sentiment
        sentiment = self.nlp.detect_sentiment(query)
        
        # Route to appropriate handler
        handlers = {
            'budgeting': self._handle_budgeting,
            'savings': self._handle_savings,
            'debt': self._handle_debt,
            'investing': self._handle_investing,
            'retirement': self._handle_retirement,
            'tax': self._handle_tax,
            'insurance': self._handle_insurance,
            'estate': self._handle_estate,
            'credit': self._handle_credit,
            'realestate': self._handle_realestate,
            'education': self._handle_education,
            'business': self._handle_business,
            'alternative': self._handle_alternative,
            'goals': self._handle_goals,
            'economy': self._handle_economy,
            'charity': self._handle_charity,
            'behavioral': self._handle_behavioral,
            'career': self._handle_career,
        }
        
        handler = handlers.get(intent, self._handle_general)
        response = handler(query, context, sentiment)
        
        # Add empathy based on sentiment
        if sentiment == 'negative':
            empathy_phrases = [
                "I understand this can be stressful. ",
                "Financial challenges are tough, but you're taking the right step by seeking guidance. ",
                "I hear your concern. ",
            ]
            response = random.choice(empathy_phrases) + response
        
        return response
    
    def _handle_budgeting(self, query: str, context: ConversationContext, sentiment: str) -> str:
        """Handle budgeting queries"""
        profile = context.user_profile
        
        # Check if we have enough info
        if profile['income'] and profile['expenses']:
            income = profile['income']
            expenses = profile['expenses']
            savings = income - expenses
            savings_rate = (savings / income) * 100
            
            response = f"Let's analyze your budget:\n\n"
            response += f"📊 Monthly Income: ${income:,.2f}\n"
            response += f"💸 Monthly Expenses: ${expenses:,.2f}\n"
            response += f"💰 Monthly Savings: ${savings:,.2f} ({savings_rate:.1f}%)\n\n"
            
            # Provide recommendations
            if savings_rate < 10:
                response += "Your savings rate is below the recommended 10-20%. Here's how to improve:\n\n"
                response += "1. Track every expense for a month to find hidden spending\n"
                response += "2. Apply the 50/30/20 rule: 50% needs, 30% wants, 20% savings\n"
                response += "3. Cut one discretionary expense category by 10%\n"
                response += f"4. If you saved just 15%, that's ${income * 0.15:,.2f}/month = ${income * 0.15 * 12:,.2f}/year!\n\n"
            elif savings_rate < 20:
                response += "You're saving a decent amount! To reach the ideal 20%:\n\n"
                response += f"• Increase savings by ${income * 0.20 - savings:,.2f}/month\n"
                response += "• Review subscriptions and recurring charges\n"
                response += "• Consider the 'pay yourself first' strategy\n\n"
            else:
                response += "Excellent savings rate! You're ahead of most people. Consider:\n\n"
                response += "• Maximizing tax-advantaged accounts (401k, IRA)\n"
                response += "• Building a 6-month emergency fund\n"
                response += "• Investing surplus for long-term growth\n\n"
            
            # Budget breakdown
            response += "Recommended budget allocation:\n"
            response += f"• Housing: ${income * 0.30:,.2f} (30%)\n"
            response += f"• Transportation: ${income * 0.15:,.2f} (15%)\n"
            response += f"• Food: ${income * 0.12:,.2f} (12%)\n"
            response += f"• Savings/Investments: ${income * 0.20:,.2f} (20%)\n"
            response += f"• Other: ${income * 0.23:,.2f} (23%)\n"
            
        else:
            response = "I'd love to help you create a budget! To give you personalized advice, I need some information:\n\n"
            if not profile['income']:
                response += "• What's your monthly income (after taxes)?\n"
            if not profile['expenses']:
                response += "• What are your total monthly expenses?\n"
            response += "\nJust share these numbers, and I'll create a detailed budget analysis for you!"
        
        return response

    
    def _handle_savings(self, query: str, context: ConversationContext, sentiment: str) -> str:
        """Handle savings queries"""
        profile = context.user_profile
        numbers = self.nlp.extract_numbers(query)
        
        response = "Let's talk about building your savings! 💰\n\n"
        
        # Emergency fund guidance
        if 'emergency' in query.lower():
            if profile['expenses']:
                monthly_expenses = profile['expenses']
                emergency_fund_target = monthly_expenses * 6
                
                response += f"Emergency Fund Target: ${emergency_fund_target:,.2f} (6 months of expenses)\n\n"
                
                if profile['savings']:
                    current_savings = profile['savings']
                    months_covered = current_savings / monthly_expenses
                    response += f"Current Savings: ${current_savings:,.2f} ({months_covered:.1f} months covered)\n\n"
                    
                    if months_covered < 3:
                        response += "⚠️ Priority: Build your emergency fund to 3 months first!\n"
                        gap = (monthly_expenses * 3) - current_savings
                        response += f"You need ${gap:,.2f} more to reach 3 months.\n\n"
                        if profile['income']:
                            months_needed = gap / (profile['income'] * 0.20)
                            response += f"Saving 20% of income (${profile['income'] * 0.20:,.2f}/month), you'll reach it in {months_needed:.1f} months.\n"
                    elif months_covered < 6:
                        response += "✓ Good start! Now work toward 6 months.\n"
                        gap = emergency_fund_target - current_savings
                        response += f"${gap:,.2f} more to reach full 6-month cushion.\n"
                    else:
                        response += "🎉 Excellent! Your emergency fund is fully funded.\n"
                        response += "Now you can focus on other goals like investing or debt payoff.\n"
                else:
                    response += "Start building your emergency fund today:\n"
                    response += "1. Open a high-yield savings account (4%+ APY)\n"
                    response += "2. Set up automatic transfers\n"
                    response += "3. Start with $1,000, then build to 3 months, then 6\n"
            else:
                response += "To calculate your emergency fund needs, what are your monthly expenses?\n"
        
        # Compound interest demonstration
        elif any(word in query.lower() for word in ['grow', 'compound', 'interest', 'calculator']):
            if len(numbers) >= 2:
                principal = numbers[0]
                years = int(numbers[1]) if len(numbers) > 1 else 10
                rate = numbers[2] / 100 if len(numbers) > 2 else Config.SAVINGS_RATE
            else:
                principal = profile.get('savings', 10000)
                years = 10
                rate = Config.SAVINGS_RATE
            
            future_value = self.calc.compound_interest(principal, rate, years)
            total_interest = future_value - principal
            
            response += f"💹 Compound Interest Projection:\n\n"
            response += f"Starting Amount: ${principal:,.2f}\n"
            response += f"Interest Rate: {rate*100:.1f}% annually\n"
            response += f"Time Period: {years} years\n"
            response += f"Future Value: ${future_value:,.2f}\n"
            response += f"Total Interest Earned: ${total_interest:,.2f}\n\n"
            
            response += "This is the power of compound interest—your money makes money!\n"
            response += "The earlier you start, the more time your money has to grow.\n\n"
            
            # Show year-by-year growth
            response += "Year-by-year growth:\n"
            for year in [1, 3, 5, 10, 20, 30]:
                if year <= years:
                    fv = self.calc.compound_interest(principal, rate, year)
                    response += f"Year {year}: ${fv:,.2f}\n"
        
        else:
            # General savings advice
            response += "Key savings strategies:\n\n"
            response += "1. **Pay Yourself First**: Automate savings before spending\n"
            response += "2. **High-Yield Savings**: Use accounts with 4%+ APY\n"
            response += "3. **Emergency Fund**: 3-6 months of expenses\n"
            response += "4. **Savings Rate**: Aim for 20% of income\n"
            response += "5. **Separate Accounts**: Different accounts for different goals\n\n"
            
            if profile['income']:
                response += f"Based on your income of ${profile['income']:,.2f}/month:\n"
                response += f"• 10% savings: ${profile['income'] * 0.10:,.2f}/month = ${profile['income'] * 0.10 * 12:,.2f}/year\n"
                response += f"• 20% savings: ${profile['income'] * 0.20:,.2f}/month = ${profile['income'] * 0.20 * 12:,.2f}/year\n"
        
        return response

    
    def _handle_debt(self, query: str, context: ConversationContext, sentiment: str) -> str:
        """Handle debt management queries"""
        profile = context.user_profile
        numbers = self.nlp.extract_numbers(query)
        
        response = "Let's tackle your debt strategically! 💪\n\n"
        
        # Debt payoff strategies
        if any(word in query.lower() for word in ['snowball', 'avalanche', 'strategy', 'method']):
            response += "**Two proven debt payoff methods:**\n\n"
            response += "🔹 **Snowball Method** (Psychological wins):\n"
            response += "• Pay minimums on all debts\n"
            response += "• Put extra money toward smallest balance\n"
            response += "• When paid off, roll that payment to next smallest\n"
            response += "• Best for: Motivation through quick wins\n\n"
            
            response += "🔹 **Avalanche Method** (Math-optimal):\n"
            response += "• Pay minimums on all debts\n"
            response += "• Put extra money toward highest interest rate\n"
            response += "• When paid off, roll to next highest rate\n"
            response += "• Best for: Saving the most money\n\n"
            
            response += "My recommendation: Avalanche saves more money, but Snowball works better if you need motivation!\n"
        
        # Loan payment calculation
        elif any(word in query.lower() for word in ['payment', 'mortgage', 'loan', 'calculate']):
            if len(numbers) >= 2:
                principal = numbers[0]
                rate = numbers[1] / 100 if numbers[1] < 1 else numbers[1] / 100
                years = numbers[2] if len(numbers) > 2 else 30
                
                monthly_payment = self.calc.loan_payment(principal, rate, years)
                total_paid = monthly_payment * years * 12
                total_interest = total_paid - principal
                
                response += f"📊 Loan Analysis:\n\n"
                response += f"Loan Amount: ${principal:,.2f}\n"
                response += f"Interest Rate: {rate*100:.2f}%\n"
                response += f"Loan Term: {years} years\n\n"
                response += f"Monthly Payment: ${monthly_payment:,.2f}\n"
                response += f"Total Amount Paid: ${total_paid:,.2f}\n"
                response += f"Total Interest: ${total_interest:,.2f}\n\n"
                
                # Show impact of extra payments
                extra_payment = 100
                extra_monthly = monthly_payment + extra_payment
                
                # Calculate payoff time with extra payment
                balance = principal
                months = 0
                monthly_rate = rate / 12
                
                while balance > 0 and months < years * 12:
                    interest = balance * monthly_rate
                    principal_paid = extra_monthly - interest
                    balance -= principal_paid
                    months += 1
                
                years_saved = (years * 12 - months) / 12
                interest_saved = total_interest - (extra_monthly * months - principal)
                
                response += f"💡 Impact of ${extra_payment} extra monthly payment:\n"
                response += f"• Payoff time: {months/12:.1f} years (save {years_saved:.1f} years!)\n"
                response += f"• Interest saved: ${interest_saved:,.2f}\n"
            else:
                response += "To calculate your loan payment, tell me:\n"
                response += "• Loan amount\n"
                response += "• Interest rate\n"
                response += "• Loan term (years)\n"
                response += "\nExample: 'Calculate payment for $300,000 at 4% for 30 years'\n"
        
        # Debt-to-income ratio
        elif profile['debt'] and profile['income']:
            dti = profile['debt_to_income']
            monthly_debt = profile['debt'] / 12  # Simplified
            
            response += f"📈 Debt-to-Income Analysis:\n\n"
            response += f"Total Debt: ${profile['debt']:,.2f}\n"
            response += f"Annual Income: ${profile['income'] * 12:,.2f}\n"
            response += f"DTI Ratio: {dti*100:.1f}%\n\n"
            
            if dti < 0.36:
                response += "✓ Healthy DTI! Lenders prefer under 36%.\n"
            elif dti < 0.43:
                response += "⚠️ Moderate DTI. You can still qualify for loans, but focus on paying down debt.\n"
            else:
                response += "🚨 High DTI! Priority: Reduce debt before taking on more.\n"
            
            response += "\nDebt reduction tips:\n"
            response += "1. Stop taking on new debt\n"
            response += "2. Pay more than minimums\n"
            response += "3. Consider debt consolidation if rates are high\n"
            response += "4. Increase income through side hustles\n"
        
        else:
            response += "General debt management principles:\n\n"
            response += "1. **List all debts**: Amount, interest rate, minimum payment\n"
            response += "2. **Always pay minimums**: Avoid late fees and credit damage\n"
            response += "3. **Attack high-interest first**: Credit cards (15-25% APR) before student loans (3-7%)\n"
            response += "4. **Consider consolidation**: If you can get a lower rate\n"
            response += "5. **Avoid new debt**: While paying off existing debt\n\n"
            response += "Remember: Debt is like a hole—stop digging before you climb out!\n"
        
        return response

    
    def _handle_investing(self, query: str, context: ConversationContext, sentiment: str) -> str:
        """Handle investing queries"""
        profile = context.user_profile
        numbers = self.nlp.extract_numbers(query)
        
        response = "Let's build your investment strategy! 📈\n\n"
        
        # Portfolio allocation
        if any(word in query.lower() for word in ['portfolio', 'allocation', 'diversif', 'asset']):
            # Use trained model for allocation recommendation
            if profile['age']:
                stock_allocation = self.model.predict_allocation(profile)
                bond_allocation = 100 - stock_allocation
                
                response += f"🎯 Recommended Asset Allocation (based on your profile):\n\n"
                response += f"Stocks/Equities: {stock_allocation:.0f}%\n"
                response += f"Bonds/Fixed Income: {bond_allocation:.0f}%\n\n"
                
                response += "This allocation is based on:\n"
                response += f"• Your age: {profile['age']} years\n"
                if profile['risk_tolerance']:
                    risk_labels = {1: 'Conservative', 2: 'Moderately Conservative', 
                                 3: 'Moderate', 4: 'Moderately Aggressive', 5: 'Aggressive'}
                    response += f"• Risk tolerance: {risk_labels.get(profile['risk_tolerance'], 'Moderate')}\n"
                if profile['years_to_retirement']:
                    response += f"• Time horizon: {profile['years_to_retirement']} years to retirement\n"
                
                response += "\n**Why this allocation?**\n"
                response += "• Stocks offer higher growth potential but more volatility\n"
                response += "• Bonds provide stability and income\n"
                response += "• Younger investors can handle more stock risk\n"
                response += "• As you age, shift toward bonds for capital preservation\n\n"
                
                # Sample portfolio
                response += "**Sample Portfolio:**\n"
                response += f"• {stock_allocation * 0.6:.0f}% U.S. Total Stock Market Index\n"
                response += f"• {stock_allocation * 0.3:.0f}% International Stock Index\n"
                response += f"• {stock_allocation * 0.1:.0f}% Emerging Markets\n"
                response += f"• {bond_allocation * 0.7:.0f}% U.S. Bond Index\n"
                response += f"• {bond_allocation * 0.3:.0f}% International Bonds\n\n"
                
                response += "Rebalance annually to maintain target allocation!\n"
            else:
                response += "To recommend a portfolio allocation, I need to know:\n"
                response += "• Your age\n"
                response += "• Your risk tolerance (1-5, where 1=conservative, 5=aggressive)\n"
                response += "• Years until retirement\n"
        
        # Investment simulation
        elif any(word in query.lower() for word in ['simulate', 'projection', 'grow', 'return']):
            if len(numbers) >= 2:
                initial = numbers[0]
                monthly = numbers[1] if len(numbers) > 1 else 0
                years = int(numbers[2]) if len(numbers) > 2 else 30
            else:
                initial = 10000
                monthly = 500
                years = 30
            
            # Run Monte Carlo simulation
            response += f"🎲 Monte Carlo Simulation ({Config.MONTE_CARLO_ITERATIONS:,} scenarios):\n\n"
            response += f"Initial Investment: ${initial:,.2f}\n"
            response += f"Monthly Contribution: ${monthly:,.2f}\n"
            response += f"Time Horizon: {years} years\n"
            response += f"Assumed Return: {Config.STOCK_RETURN*100:.0f}% ± 15% (stock-like volatility)\n\n"
            
            results = self.calc.monte_carlo_simulation(
                initial, monthly * 12, years, Config.STOCK_RETURN, 0.15
            )
            
            response += "**Projected Outcomes:**\n"
            response += f"• Median (50th percentile): ${results['median']:,.2f}\n"
            response += f"• Conservative (25th percentile): ${results['percentile_25']:,.2f}\n"
            response += f"• Optimistic (75th percentile): ${results['percentile_75']:,.2f}\n"
            response += f"• Best case (90th percentile): ${results['percentile_90']:,.2f}\n"
            response += f"• Worst case (10th percentile): ${results['percentile_10']:,.2f}\n\n"
            
            total_contributions = initial + (monthly * 12 * years)
            median_gain = results['median'] - total_contributions
            
            response += f"Total Contributions: ${total_contributions:,.2f}\n"
            response += f"Median Investment Gain: ${median_gain:,.2f}\n\n"
            
            response += "**Key Insights:**\n"
            response += "• There's a wide range of possible outcomes due to market volatility\n"
            response += "• The median represents the most likely scenario\n"
            response += "• Long-term investing smooths out short-term fluctuations\n"
            response += "• Regular contributions (dollar-cost averaging) reduce timing risk\n"
        
        else:
            # General investing education
            response += "**Investment Fundamentals:**\n\n"
            response += "1. **Start Early**: Time in the market beats timing the market\n"
            response += "2. **Diversify**: Don't put all eggs in one basket\n"
            response += "3. **Low Costs**: Index funds typically beat active management\n"
            response += "4. **Stay Invested**: Don't panic sell during downturns\n"
            response += "5. **Rebalance**: Maintain your target allocation\n\n"
            
            response += "**Asset Classes Explained:**\n\n"
            response += "📊 **Stocks**: Ownership in companies\n"
            response += "   • Higher risk, higher potential return (~7-10% historically)\n"
            response += "   • Best for long-term goals (10+ years)\n\n"
            
            response += "📈 **Bonds**: Loans to governments/corporations\n"
            response += "   • Lower risk, lower return (~3-5%)\n"
            response += "   • Provide stability and income\n\n"
            
            response += "🏠 **Real Estate**: Property investments\n"
            response += "   • Can provide income and appreciation\n"
            response += "   • REITs offer real estate exposure without buying property\n\n"
            
            response += "💰 **Index Funds/ETFs**: Baskets of stocks/bonds\n"
            response += "   • Instant diversification\n"
            response += "   • Low fees\n"
            response += "   • My top recommendation for most investors!\n"
        
        return response

    
    def _handle_retirement(self, query: str, context: ConversationContext, sentiment: str) -> str:
        """Handle retirement planning queries"""
        profile = context.user_profile
        numbers = self.nlp.extract_numbers(query)
        
        response = "Let's plan for your retirement! 🏖️\n\n"
        
        if profile['age'] and profile['expenses']:
            current_age = profile['age']
            retirement_age = 65
            
            # Extract retirement age from query if mentioned
            if 'retire at' in query.lower() or 'retirement age' in query.lower():
                for num in numbers:
                    if 50 <= num <= 75:
                        retirement_age = int(num)
                        break
            
            annual_expenses = profile['expenses'] * 12
            
            # Calculate retirement needs
            ret_calc = self.calc.retirement_needs(
                current_age, retirement_age, annual_expenses
            )
            
            response += f"🎯 Retirement Analysis:\n\n"
            response += f"Current Age: {current_age}\n"
            response += f"Target Retirement Age: {retirement_age}\n"
            response += f"Years to Retirement: {ret_calc['years_to_retirement']}\n\n"
            
            response += f"Current Annual Expenses: ${ret_calc['current_annual_expenses']:,.2f}\n"
            response += f"Estimated Expenses at Retirement: ${ret_calc['future_annual_expenses']:,.2f}\n"
            response += f"  (adjusted for {Config.INFLATION_RATE*100:.1f}% inflation)\n\n"
            
            response += f"**Nest Egg Needed: ${ret_calc['nest_egg_needed']:,.2f}**\n"
            response += f"  (using {ret_calc['withdrawal_rate']*100:.0f}% withdrawal rate)\n\n"
            
            # Calculate required monthly savings
            if profile['savings']:
                current_savings = profile['savings']
                gap = ret_calc['nest_egg_needed'] - current_savings
                
                response += f"Current Savings: ${current_savings:,.2f}\n"
                response += f"Gap to Fill: ${gap:,.2f}\n\n"
                
                if gap > 0:
                    # Calculate required monthly contribution
                    years = ret_calc['years_to_retirement']
                    future_value_current = self.calc.compound_interest(
                        current_savings, Config.STOCK_RETURN, years
                    )
                    
                    remaining_needed = ret_calc['nest_egg_needed'] - future_value_current
                    
                    if remaining_needed > 0:
                        # Calculate monthly payment needed
                        monthly_rate = Config.STOCK_RETURN / 12
                        months = years * 12
                        
                        if monthly_rate > 0:
                            monthly_needed = remaining_needed * monthly_rate / \
                                           (((1 + monthly_rate) ** months - 1))
                        else:
                            monthly_needed = remaining_needed / months
                        
                        response += f"**Action Plan:**\n"
                        response += f"• Your current ${current_savings:,.2f} will grow to ${future_value_current:,.2f}\n"
                        response += f"• You need to save ${monthly_needed:,.2f}/month\n"
                        response += f"• At {Config.STOCK_RETURN*100:.0f}% return for {years} years\n\n"
                        
                        if profile['income']:
                            savings_rate_needed = (monthly_needed / profile['income']) * 100
                            response += f"That's {savings_rate_needed:.1f}% of your monthly income.\n\n"
                    else:
                        response += "🎉 Great news! Your current savings are on track!\n\n"
                else:
                    response += "🎉 Congratulations! You've already reached your retirement goal!\n\n"
            else:
                # No current savings
                years = ret_calc['years_to_retirement']
                monthly_rate = Config.STOCK_RETURN / 12
                months = years * 12
                
                monthly_needed = ret_calc['nest_egg_needed'] * monthly_rate / \
                               (((1 + monthly_rate) ** months - 1))
                
                response += f"Starting from $0, you'd need to save:\n"
                response += f"• ${monthly_needed:,.2f} per month\n"
                response += f"• ${monthly_needed * 12:,.2f} per year\n"
                response += f"• For {years} years at {Config.STOCK_RETURN*100:.0f}% return\n\n"
            
            response += "**Retirement Accounts to Use:**\n"
            response += "1. **401(k)**: Employer match is free money! Contribute at least enough to get full match\n"
            response += "2. **IRA/Roth IRA**: $6,500/year limit ($7,500 if 50+)\n"
            response += "3. **HSA**: Triple tax advantage if you have a high-deductible health plan\n"
            response += "4. **Taxable brokerage**: After maxing tax-advantaged accounts\n\n"
            
            response += "**The 4% Rule:**\n"
            response += "Withdraw 4% of your nest egg annually in retirement.\n"
            response += "This gives a high probability your money will last 30+ years.\n"
            
        else:
            response += "To create your retirement plan, I need:\n"
            if not profile['age']:
                response += "• Your current age\n"
            if not profile['expenses']:
                response += "• Your monthly expenses\n"
            response += "\nOptionally:\n"
            response += "• Target retirement age (default: 65)\n"
            response += "• Current retirement savings\n"
        
        return response

    
    def _handle_tax(self, query: str, context: ConversationContext, sentiment: str) -> str:
        """Handle tax planning queries"""
        profile = context.user_profile
        numbers = self.nlp.extract_numbers(query)
        
        response = "Let's optimize your tax strategy! 💼\n\n"
        
        if profile['income']:
            annual_income = profile['income'] * 12
            filing_status = 'single'  # Default
            
            if 'married' in query.lower():
                filing_status = 'married'
            
            tax_calc = self.calc.tax_bracket_calculation(annual_income, filing_status)
            
            response += f"📊 Tax Analysis ({filing_status.title()} Filing):\n\n"
            response += f"Gross Income: ${tax_calc['gross_income']:,.2f}\n"
            response += f"Estimated Federal Tax: ${tax_calc['total_tax']:,.2f}\n"
            response += f"After-Tax Income: ${tax_calc['after_tax_income']:,.2f}\n\n"
            response += f"Effective Tax Rate: {tax_calc['effective_rate']*100:.1f}%\n"
            response += f"Marginal Tax Rate: {tax_calc['marginal_rate']*100:.0f}%\n\n"
            
            response += "**What this means:**\n"
            response += f"• You pay {tax_calc['effective_rate']*100:.1f}% of total income in taxes (effective rate)\n"
            response += f"• Each additional dollar earned is taxed at {tax_calc['marginal_rate']*100:.0f}% (marginal rate)\n\n"
            
            response += "**Tax Reduction Strategies:**\n\n"
            
            # 401(k) contribution impact
            contribution_401k = 22500  # 2023 limit
            reduced_income = annual_income - contribution_401k
            if reduced_income > 0:
                new_tax = self.calc.tax_bracket_calculation(reduced_income, filing_status)
                tax_savings = tax_calc['total_tax'] - new_tax['total_tax']
                
                response += f"1. **Max out 401(k)** (${contribution_401k:,.0f}/year):\n"
                response += f"   • Reduces taxable income to ${reduced_income:,.2f}\n"
                response += f"   • Saves ${tax_savings:,.2f} in taxes\n"
                response += f"   • Effective cost: ${contribution_401k - tax_savings:,.2f}\n\n"
            
            # IRA contribution
            ira_limit = 6500
            response += f"2. **Traditional IRA** (${ira_limit:,.0f}/year):\n"
            response += f"   • Tax deduction reduces current taxes\n"
            response += f"   • Saves ~${ira_limit * tax_calc['marginal_rate']:,.2f}\n\n"
            
            response += "3. **HSA** (if eligible):\n"
            response += "   • Triple tax advantage: Deductible, grows tax-free, withdrawals tax-free for medical\n"
            response += "   • $3,850 individual / $7,750 family (2023)\n\n"
            
            response += "4. **Tax-Loss Harvesting**:\n"
            response += "   • Sell losing investments to offset gains\n"
            response += "   • Can deduct up to $3,000 in losses against ordinary income\n\n"
            
            response += "5. **Charitable Donations**:\n"
            response += "   • Deductible if you itemize\n"
            response += "   • Donate appreciated stock to avoid capital gains\n\n"
            
            # Standard deduction info
            standard_deduction = 13850 if filing_status == 'single' else 27700
            response += f"**Standard Deduction ({filing_status}):** ${standard_deduction:,.0f}\n"
            response += "Itemize only if deductions exceed this amount.\n"
            
        else:
            response += "To analyze your tax situation, I need your annual income.\n\n"
            response += "**General Tax Tips:**\n"
            response += "• Contribute to tax-advantaged accounts (401k, IRA, HSA)\n"
            response += "• Time capital gains for lower tax years\n"
            response += "• Keep good records of deductible expenses\n"
            response += "• Consider tax-loss harvesting\n"
            response += "• Understand the difference between marginal and effective rates\n"
        
        response += "\n⚠️ Disclaimer: This is general information. Consult a tax professional for your specific situation.\n"
        
        return response
    
    def _handle_insurance(self, query: str, context: ConversationContext, sentiment: str) -> str:
        """Handle insurance queries"""
        response = "Let's review your insurance needs! 🛡️\n\n"
        
        response += "Insurance is your financial safety net. Here's what you need:\n\n"
        
        response += "**1. Health Insurance** (Essential)\n"
        response += "• Protects against catastrophic medical costs\n"
        response += "• Get through employer, marketplace, or Medicaid\n"
        response += "• Consider HSA-eligible high-deductible plans if healthy\n\n"
        
        response += "**2. Life Insurance** (If others depend on your income)\n"
        response += "• Term life: Cheaper, covers specific period (20-30 years)\n"
        response += "• Whole life: More expensive, permanent coverage\n"
        response += "• Rule of thumb: 10-12x your annual income\n"
        if context.user_profile['income']:
            coverage = context.user_profile['income'] * 12 * 10
            response += f"• For you: ~${coverage:,.0f} coverage\n"
        response += "\n"
        
        response += "**3. Disability Insurance** (Protect your income)\n"
        response += "• Replaces 60-70% of income if you can't work\n"
        response += "• Check if employer offers it\n"
        response += "• Especially important if you're the primary earner\n\n"
        
        response += "**4. Auto Insurance** (Required by law)\n"
        response += "• Liability: Covers damage you cause\n"
        response += "• Collision/Comprehensive: Covers your vehicle\n"
        response += "• Increase deductible to lower premiums\n\n"
        
        response += "**5. Homeowners/Renters Insurance**\n"
        response += "• Homeowners: Required by mortgage lender\n"
        response += "• Renters: Cheap (~$15/month) and protects your stuff\n\n"
        
        response += "**6. Umbrella Policy** (For high net worth)\n"
        response += "• Extra liability coverage beyond auto/home\n"
        response += "• $1M coverage costs ~$200/year\n"
        response += "• Consider if net worth > $500k\n\n"
        
        response += "**Insurance Tips:**\n"
        response += "• Shop around every 2-3 years\n"
        response += "• Bundle policies for discounts\n"
        response += "• Higher deductibles = lower premiums\n"
        response += "• Don't over-insure or under-insure\n"
        response += "• Review coverage after major life events\n"
        
        return response

    
    def _handle_estate(self, query: str, context: ConversationContext, sentiment: str) -> str:
        """Handle estate planning queries"""
        response = "Let's plan your legacy! 📜\n\n"
        
        response += "Estate planning ensures your wishes are honored and loved ones are protected.\n\n"
        
        response += "**Essential Documents:**\n\n"
        
        response += "1. **Will**\n"
        response += "   • Specifies how assets are distributed\n"
        response += "   • Names guardian for minor children\n"
        response += "   • Everyone 18+ should have one\n"
        response += "   • Update after major life events\n\n"
        
        response += "2. **Living Trust** (Optional but useful)\n"
        response += "   • Avoids probate (faster, private)\n"
        response += "   • Maintains control during your lifetime\n"
        response += "   • Consider if estate > $100k\n\n"
        
        response += "3. **Power of Attorney (POA)**\n"
        response += "   • Financial POA: Manages finances if you're incapacitated\n"
        response += "   • Healthcare POA: Makes medical decisions\n"
        response += "   • Critical for everyone\n\n"
        
        response += "4. **Healthcare Directive / Living Will**\n"
        response += "   • Specifies end-of-life care preferences\n"
        response += "   • Reduces burden on family\n\n"
        
        response += "5. **Beneficiary Designations**\n"
        response += "   • Review on all accounts (401k, IRA, life insurance)\n"
        response += "   • These override your will!\n"
        response += "   • Update after marriage, divorce, births\n\n"
        
        response += "**Estate Tax Considerations:**\n"
        response += "• Federal estate tax exemption: $12.92M per person (2023)\n"
        response += "• Most estates won't owe federal tax\n"
        response += "• Some states have lower thresholds\n"
        response += "• Married couples can combine exemptions\n\n"
        
        response += "**Wealth Transfer Strategies:**\n"
        response += "• Annual gift exclusion: $17,000 per person (2023)\n"
        response += "• 529 plans for education\n"
        response += "• Irrevocable trusts for tax benefits\n"
        response += "• Charitable remainder trusts\n\n"
        
        response += "**Action Steps:**\n"
        response += "1. Create or update your will\n"
        response += "2. Designate POA and healthcare proxy\n"
        response += "3. Review all beneficiary designations\n"
        response += "4. Consider trust if estate is complex\n"
        response += "5. Communicate plans with family\n"
        response += "6. Store documents securely and tell someone where\n\n"
        
        response += "💡 Tip: Use an estate planning attorney for complex situations.\n"
        response += "Simple wills can be done online for $100-300.\n"
        
        return response
    
    def _handle_credit(self, query: str, context: ConversationContext, sentiment: str) -> str:
        """Handle credit score queries"""
        profile = context.user_profile
        response = "Let's improve your credit! 💳\n\n"
        
        if profile['credit_score']:
            score = profile['credit_score']
            response += f"Your Credit Score: {score}\n\n"
            
            if score >= 800:
                rating = "Exceptional"
                advice = "You have excellent credit! Maintain your good habits."
            elif score >= 740:
                rating = "Very Good"
                advice = "Great credit! You qualify for the best rates."
            elif score >= 670:
                rating = "Good"
                advice = "Solid credit. Keep improving to access better rates."
            elif score >= 580:
                rating = "Fair"
                advice = "Focus on improving to access better loan terms."
            else:
                rating = "Poor"
                advice = "Priority: Rebuild your credit with the steps below."
            
            response += f"Rating: {rating}\n"
            response += f"{advice}\n\n"
        
        response += "**Credit Score Factors:**\n\n"
        response += "1. **Payment History (35%)**\n"
        response += "   • Most important factor\n"
        response += "   • Pay all bills on time, every time\n"
        response += "   • Set up autopay to never miss\n\n"
        
        response += "2. **Credit Utilization (30%)**\n"
        response += "   • Keep below 30% of credit limits\n"
        response += "   • Below 10% is ideal\n"
        response += "   • Pay down balances or request limit increases\n\n"
        
        response += "3. **Credit History Length (15%)**\n"
        response += "   • Older accounts are better\n"
        response += "   • Don't close old cards\n"
        response += "   • Average age of accounts matters\n\n"
        
        response += "4. **Credit Mix (10%)**\n"
        response += "   • Different types: credit cards, loans, mortgage\n"
        response += "   • Not critical, but helps\n\n"
        
        response += "5. **New Credit (10%)**\n"
        response += "   • Hard inquiries lower score temporarily\n"
        response += "   • Don't apply for multiple cards at once\n\n"
        
        response += "**How to Improve Your Score:**\n\n"
        response += "✓ Pay all bills on time (set up autopay)\n"
        response += "✓ Pay down credit card balances\n"
        response += "✓ Don't close old credit cards\n"
        response += "✓ Request credit limit increases\n"
        response += "✓ Dispute errors on credit report\n"
        response += "✓ Become authorized user on someone's good account\n"
        response += "✓ Use credit monitoring (free from many sources)\n\n"
        
        response += "**What Hurts Your Score:**\n\n"
        response += "✗ Late payments (stay for 7 years!)\n"
        response += "✗ High credit utilization\n"
        response += "✗ Collections and charge-offs\n"
        response += "✗ Bankruptcy (stays 7-10 years)\n"
        response += "✗ Too many hard inquiries\n\n"
        
        response += "**Free Credit Reports:**\n"
        response += "• AnnualCreditReport.com (official site)\n"
        response += "• Check all 3 bureaus: Equifax, Experian, TransUnion\n"
        response += "• Review for errors and dispute if needed\n\n"
        
        response += "💡 Building credit takes time. Focus on consistent good habits!\n"
        
        return response

    
    def _handle_realestate(self, query: str, context: ConversationContext, sentiment: str) -> str:
        """Handle real estate queries"""
        profile = context.user_profile
        numbers = self.nlp.extract_numbers(query)
        response = "Let's explore real estate! 🏠\n\n"
        
        # Rent vs Buy analysis
        if 'rent' in query.lower() and 'buy' in query.lower():
            response += "**Rent vs. Buy Decision:**\n\n"
            response += "**Buy if:**\n"
            response += "• You plan to stay 5+ years\n"
            response += "• You have 20% down payment saved\n"
            response += "• Monthly payment < 28% of gross income\n"
            response += "• You want to build equity\n"
            response += "• You value stability and control\n\n"
            
            response += "**Rent if:**\n"
            response += "• You might move in < 5 years\n"
            response += "• You don't have down payment\n"
            response += "• You want flexibility\n"
            response += "• Local market is overpriced\n"
            response += "• You prefer not to handle maintenance\n\n"
            
            response += "**Hidden Costs of Homeownership:**\n"
            response += "• Property taxes (1-2% of home value annually)\n"
            response += "• Insurance ($1,000-2,000/year)\n"
            response += "• Maintenance (1% of home value annually)\n"
            response += "• HOA fees (if applicable)\n"
            response += "• Closing costs (2-5% of purchase price)\n"
        
        # Mortgage calculation
        elif 'mortgage' in query.lower() or 'payment' in query.lower():
            if len(numbers) >= 2:
                home_price = numbers[0]
                down_payment_pct = numbers[1] / 100 if numbers[1] < 1 else numbers[1] / 100
                interest_rate = numbers[2] / 100 if len(numbers) > 2 else 0.065
                
                down_payment = home_price * down_payment_pct
                loan_amount = home_price - down_payment
                
                monthly_payment = self.calc.loan_payment(loan_amount, interest_rate, 30)
                
                # Add property tax and insurance estimates
                annual_property_tax = home_price * 0.012
                annual_insurance = 1500
                monthly_tax_insurance = (annual_property_tax + annual_insurance) / 12
                
                total_monthly = monthly_payment + monthly_tax_insurance
                
                response += f"🏡 Home Purchase Analysis:\n\n"
                response += f"Home Price: ${home_price:,.2f}\n"
                response += f"Down Payment ({down_payment_pct*100:.0f}%): ${down_payment:,.2f}\n"
                response += f"Loan Amount: ${loan_amount:,.2f}\n"
                response += f"Interest Rate: {interest_rate*100:.2f}%\n"
                response += f"Loan Term: 30 years\n\n"
                
                response += f"**Monthly Costs:**\n"
                response += f"Principal & Interest: ${monthly_payment:,.2f}\n"
                response += f"Property Tax (est): ${annual_property_tax/12:,.2f}\n"
                response += f"Insurance (est): ${annual_insurance/12:,.2f}\n"
                response += f"**Total Monthly: ${total_monthly:,.2f}**\n\n"
                
                if profile['income']:
                    housing_ratio = (total_monthly / profile['income']) * 100
                    response += f"Housing Expense Ratio: {housing_ratio:.1f}% of income\n"
                    if housing_ratio <= 28:
                        response += "✓ Within recommended 28% guideline\n"
                    else:
                        response += "⚠️ Above recommended 28% guideline\n"
                    response += "\n"
                
                # Total cost over 30 years
                total_paid = monthly_payment * 360
                total_interest = total_paid - loan_amount
                
                response += f"**30-Year Totals:**\n"
                response += f"Total Paid: ${total_paid:,.2f}\n"
                response += f"Total Interest: ${total_interest:,.2f}\n"
                response += f"Total Cost (with down payment): ${total_paid + down_payment:,.2f}\n"
            else:
                response += "To calculate mortgage payment, tell me:\n"
                response += "• Home price\n"
                response += "• Down payment percentage\n"
                response += "• Interest rate (optional)\n"
                response += "\nExample: 'Mortgage for $400,000 with 20% down at 6.5%'\n"
        
        # Affordability
        elif 'afford' in query.lower():
            if profile['income']:
                monthly_income = profile['income']
                max_housing = monthly_income * 0.28
                
                # Estimate affordable home price
                # Assume 20% down, 6.5% rate, 30 years
                # Work backwards from max payment
                interest_rate = 0.065
                loan_term = 30
                
                # Max loan amount based on payment
                monthly_rate = interest_rate / 12
                num_payments = loan_term * 12
                
                # Subtract estimated tax/insurance from max payment
                estimated_tax_ins = max_housing * 0.25  # Rough estimate
                max_pi_payment = max_housing - estimated_tax_ins
                
                # Calculate max loan amount
                max_loan = max_pi_payment * ((1 - (1 + monthly_rate) ** -num_payments) / monthly_rate)
                
                # Add down payment (assume 20%)
                max_home_price = max_loan / 0.80
                
                response += f"💰 Home Affordability Analysis:\n\n"
                response += f"Monthly Income: ${monthly_income:,.2f}\n"
                response += f"Max Housing Payment (28% rule): ${max_housing:,.2f}\n\n"
                
                response += f"**Estimated Affordable Home Price: ${max_home_price:,.2f}**\n"
                response += f"(Assuming 20% down, 6.5% rate, 30-year mortgage)\n\n"
                
                response += f"This would require:\n"
                response += f"• Down Payment: ${max_home_price * 0.20:,.2f}\n"
                response += f"• Closing Costs: ${max_home_price * 0.03:,.2f} (est 3%)\n"
                response += f"• Total Cash Needed: ${max_home_price * 0.23:,.2f}\n"
            else:
                response += "To calculate home affordability, I need your monthly income.\n"
        
        else:
            response += "**Real Estate Investment Basics:**\n\n"
            response += "**Primary Residence:**\n"
            response += "• Builds equity over time\n"
            response += "• Mortgage interest deduction (if you itemize)\n"
            response += "• Capital gains exclusion: $250k single / $500k married\n\n"
            
            response += "**Rental Property:**\n"
            response += "• Generates passive income\n"
            response += "• Tax benefits: depreciation, expense deductions\n"
            response += "• Requires management and maintenance\n"
            response += "• 1% rule: Monthly rent should be 1% of purchase price\n\n"
            
            response += "**REITs (Real Estate Investment Trusts):**\n"
            response += "• Own real estate without buying property\n"
            response += "• Liquid (trade like stocks)\n"
            response += "• Dividend income\n"
            response += "• Good for diversification\n"
        
        return response

    
    def _handle_education(self, query: str, context: ConversationContext, sentiment: str) -> str:
        """Handle education funding queries"""
        response = "Let's plan for education costs! 🎓\n\n"
        
        response += "**529 College Savings Plan** (Best option for most):\n\n"
        response += "**Benefits:**\n"
        response += "• Tax-free growth and withdrawals for education\n"
        response += "• High contribution limits ($300k+ lifetime)\n"
        response += "• State tax deduction in many states\n"
        response += "• Can change beneficiary to another family member\n"
        response += "• Can be used for K-12 tuition ($10k/year)\n\n"
        
        response += "**Drawbacks:**\n"
        response += "• 10% penalty + taxes if used for non-education\n"
        response += "• Counts as parent asset for financial aid (minimal impact)\n\n"
        
        # College cost projection
        current_cost = 30000  # Average annual cost
        years = 18
        inflation = 0.05  # Higher than general inflation
        
        future_cost = current_cost * (1 + inflation) ** years
        total_4_years = future_cost * 4
        
        response += f"**College Cost Projection:**\n"
        response += f"Current average annual cost: ${current_cost:,.0f}\n"
        response += f"Projected cost in {years} years: ${future_cost:,.0f}/year\n"
        response += f"Total 4-year cost: ${total_4_years:,.0f}\n"
        response += f"(Assuming {inflation*100:.0f}% annual increase)\n\n"
        
        # Savings calculation
        monthly_rate = 0.07 / 12
        months = years * 12
        monthly_needed = total_4_years * monthly_rate / (((1 + monthly_rate) ** months - 1))
        
        response += f"**To save ${total_4_years:,.0f} in {years} years:**\n"
        response += f"Save ${monthly_needed:,.2f}/month at 7% return\n\n"
        
        response += "**Other Education Savings Options:**\n\n"
        
        response += "**Coverdell ESA:**\n"
        response += "• $2,000/year contribution limit\n"
        response += "• Tax-free growth for education\n"
        response += "• Can use for K-12 expenses\n"
        response += "• Income limits apply\n\n"
        
        response += "**Custodial Accounts (UGMA/UTMA):**\n"
        response += "• No contribution limits\n"
        response += "• Child gains control at 18-21\n"
        response += "• Taxed at child's rate (kiddie tax applies)\n"
        response += "• Counts heavily against financial aid\n\n"
        
        response += "**Roth IRA:**\n"
        response += "• Can withdraw contributions anytime tax-free\n"
        response += "• Earnings can be used for education (with conditions)\n"
        response += "• Doesn't count for financial aid\n"
        response += "• Dual purpose: retirement backup\n\n"
        
        response += "**Financial Aid Tips:**\n"
        response += "• Fill out FAFSA every year\n"
        response += "• Parent assets count less than student assets\n"
        response += "• Grandparent 529s don't count until distributed\n"
        response += "• Apply for scholarships aggressively\n\n"
        
        response += "**Student Loan Guidance:**\n"
        response += "• Federal loans first (better terms, protections)\n"
        response += "• Borrow only what you need\n"
        response += "• Rule of thumb: Don't borrow more than expected first-year salary\n"
        response += "• Consider income-driven repayment plans\n"
        
        return response
    
    def _handle_business(self, query: str, context: ConversationContext, sentiment: str) -> str:
        """Handle business finance queries"""
        response = "Let's talk business finance! 💼\n\n"
        
        response += "**Starting a Business - Financial Checklist:**\n\n"
        
        response += "**1. Startup Costs:**\n"
        response += "• Equipment and supplies\n"
        response += "• Legal and licensing fees\n"
        response += "• Initial inventory\n"
        response += "• Marketing and branding\n"
        response += "• Website and technology\n"
        response += "• 6 months of operating expenses\n\n"
        
        response += "**2. Business Structure:**\n"
        response += "• Sole Proprietorship: Simplest, but no liability protection\n"
        response += "• LLC: Liability protection, flexible taxes\n"
        response += "• S-Corp: Tax benefits if profitable\n"
        response += "• C-Corp: For high-growth startups seeking investment\n\n"
        
        response += "**3. Separate Finances:**\n"
        response += "• Open business bank account\n"
        response += "• Get business credit card\n"
        response += "• Never mix personal and business money\n"
        response += "• Makes accounting and taxes much easier\n\n"
        
        response += "**4. Cash Flow Management:**\n"
        response += "• Track every dollar in and out\n"
        response += "• Maintain 3-6 months operating expenses\n"
        response += "• Invoice promptly and follow up\n"
        response += "• Negotiate payment terms with vendors\n"
        response += "• Use accounting software (QuickBooks, FreshBooks)\n\n"
        
        response += "**5. Pricing Strategy:**\n"
        response += "• Calculate true cost (materials + time + overhead)\n"
        response += "• Add profit margin (30-50% for products, 50-100% for services)\n"
        response += "• Research competitor pricing\n"
        response += "• Don't undervalue your work\n\n"
        
        response += "**6. Break-Even Analysis:**\n"
        response += "• Fixed costs: Rent, insurance, salaries\n"
        response += "• Variable costs: Materials, commissions\n"
        response += "• Break-even = Fixed Costs / (Price - Variable Cost per Unit)\n\n"
        
        response += "**7. Funding Options:**\n"
        response += "• Bootstrap: Self-fund (maintains control)\n"
        response += "• Small business loan: SBA loans, bank loans\n"
        response += "• Business line of credit: Flexible borrowing\n"
        response += "• Angel investors: Early-stage funding\n"
        response += "• Venture capital: High-growth potential\n"
        response += "• Crowdfunding: Kickstarter, Indiegogo\n\n"
        
        response += "**8. Tax Considerations:**\n"
        response += "• Set aside 25-30% of income for taxes\n"
        response += "• Make quarterly estimated tax payments\n"
        response += "• Deduct business expenses\n"
        response += "• Home office deduction if applicable\n"
        response += "• Hire a CPA for tax planning\n\n"
        
        response += "**9. Key Metrics to Track:**\n"
        response += "• Revenue and profit margins\n"
        response += "• Customer acquisition cost\n"
        response += "• Lifetime customer value\n"
        response += "• Cash runway (months until money runs out)\n"
        response += "• Accounts receivable aging\n\n"
        
        response += "💡 Most businesses fail due to cash flow problems, not lack of profit!\n"
        
        return response

    
    def _handle_alternative(self, query: str, context: ConversationContext, sentiment: str) -> str:
        """Handle alternative investment queries"""
        response = "Let's discuss alternative investments! ⚠️\n\n"
        
        response += "**Important Warning:** Alternative investments are high-risk and speculative.\n"
        response += "Only invest money you can afford to lose completely.\n\n"
        
        if 'crypto' in query.lower() or 'bitcoin' in query.lower():
            response += "**Cryptocurrency:**\n\n"
            response += "**Pros:**\n"
            response += "• High growth potential\n"
            response += "• Decentralized\n"
            response += "• 24/7 trading\n"
            response += "• Portfolio diversification\n\n"
            
            response += "**Cons:**\n"
            response += "• Extreme volatility (50%+ swings)\n"
            response += "• Regulatory uncertainty\n"
            response += "• Security risks (hacks, lost keys)\n"
            response += "• No intrinsic value or cash flow\n"
            response += "• Tax complexity\n\n"
            
            response += "**If You Invest in Crypto:**\n"
            response += "• Limit to 1-5% of portfolio\n"
            response += "• Use reputable exchanges (Coinbase, Kraken)\n"
            response += "• Consider hardware wallet for large amounts\n"
            response += "• Never invest more than you can lose\n"
            response += "• Understand it's speculation, not investing\n"
            response += "• Track all transactions for taxes\n\n"
        
        response += "**Other Alternative Investments:**\n\n"
        
        response += "**Peer-to-Peer Lending:**\n"
        response += "• Lend money directly to borrowers\n"
        response += "• Returns: 5-10% (but with default risk)\n"
        response += "• Platforms: LendingClub, Prosper\n"
        response += "• Risk: Borrower defaults\n\n"
        
        response += "**Commodities (Gold, Silver, Oil):**\n"
        response += "• Hedge against inflation\n"
        response += "• No cash flow or dividends\n"
        response += "• High volatility\n"
        response += "• Consider commodity ETFs instead of physical\n\n"
        
        response += "**Collectibles (Art, Wine, Cards):**\n"
        response += "• Requires expertise\n"
        response += "• Illiquid (hard to sell quickly)\n"
        response += "• Storage and insurance costs\n"
        response += "• Buy for enjoyment, not investment\n\n"
        
        response += "**Private Equity / Venture Capital:**\n"
        response += "• High minimum investments ($25k-$1M+)\n"
        response += "• Long lock-up periods (7-10 years)\n"
        response += "• High risk, high potential return\n"
        response += "• Only for accredited investors\n\n"
        
        response += "**My Recommendation:**\n"
        response += "For 95% of people, a simple portfolio of low-cost index funds is better than alternatives.\n"
        response += "If you want alternatives, limit to <10% of portfolio and only after you have:\n"
        response += "• Emergency fund\n"
        response += "• No high-interest debt\n"
        response += "• Maxed retirement accounts\n"
        response += "• Solid core portfolio\n\n"
        
        response += "🚨 Avoid: Get-rich-quick schemes, MLMs, penny stocks, forex trading, NFTs (unless you deeply understand them)\n"
        
        return response
    
    def _handle_goals(self, query: str, context: ConversationContext, sentiment: str) -> str:
        """Handle financial goal setting"""
        response = "Let's set and track your financial goals! 🎯\n\n"
        
        response += "**SMART Goal Framework:**\n\n"
        response += "• **S**pecific: Clearly defined\n"
        response += "• **M**easurable: Track progress\n"
        response += "• **A**chievable: Realistic given your situation\n"
        response += "• **R**elevant: Aligned with your values\n"
        response += "• **T**ime-bound: Has a deadline\n\n"
        
        response += "**Common Financial Goals:**\n\n"
        
        response += "**Short-term (< 1 year):**\n"
        response += "• Build $1,000 emergency fund\n"
        response += "• Pay off credit card\n"
        response += "• Save for vacation\n"
        response += "• Create and stick to budget\n\n"
        
        response += "**Medium-term (1-5 years):**\n"
        response += "• Save for home down payment\n"
        response += "• Pay off student loans\n"
        response += "• Build 6-month emergency fund\n"
        response += "• Save for wedding\n"
        response += "• Buy a car with cash\n\n"
        
        response += "**Long-term (5+ years):**\n"
        response += "• Retirement savings\n"
        response += "• Children's college fund\n"
        response += "• Pay off mortgage\n"
        response += "• Financial independence\n"
        response += "• Start a business\n\n"
        
        response += "**Goal Prioritization:**\n\n"
        response += "1. **Foundation**: Emergency fund + high-interest debt\n"
        response += "2. **Retirement**: Take advantage of employer match\n"
        response += "3. **Other goals**: Based on timeline and importance\n"
        response += "4. **Extra**: Additional retirement, taxable investing\n\n"
        
        response += "**Tracking Progress:**\n\n"
        response += "• Set specific milestones\n"
        response += "• Review monthly\n"
        response += "• Celebrate small wins\n"
        response += "• Adjust as life changes\n"
        response += "• Use visual trackers (charts, apps)\n\n"
        
        response += "**Example Goal:**\n"
        response += "❌ Bad: 'Save more money'\n"
        response += "✓ Good: 'Save $20,000 for home down payment by December 2025 by saving $833/month'\n\n"
        
        response += "What specific goal would you like to work on? I can help you create a plan!\n"
        
        return response
    
    def _handle_economy(self, query: str, context: ConversationContext, sentiment: str) -> str:
        """Handle economy and macro finance queries"""
        response = "Let's discuss economic factors! 📊\n\n"
        
        response += "**Key Economic Concepts:**\n\n"
        
        response += "**Inflation:**\n"
        response += "• Rising prices over time\n"
        response += "• Erodes purchasing power\n"
        response += "• Historical average: 2-3% annually\n"
        response += "• Combat with: Investing, raises, efficiency\n"
        response += "• Assets that hedge: Stocks, real estate, I-bonds\n\n"
        
        response += "**Interest Rates:**\n"
        response += "• Set by Federal Reserve\n"
        response += "• Higher rates: Borrowing costs more, savings earn more\n"
        response += "• Lower rates: Cheaper loans, lower savings returns\n"
        response += "• Affects: Mortgages, car loans, credit cards, bonds\n\n"
        
        response += "**Economic Cycles:**\n"
        response += "• Expansion: Growth, low unemployment, rising prices\n"
        response += "• Peak: Maximum growth\n"
        response += "• Contraction/Recession: Declining growth, rising unemployment\n"
        response += "• Trough: Bottom, before recovery\n\n"
        
        response += "**Recession Preparation:**\n"
        response += "• Build larger emergency fund (6-12 months)\n"
        response += "• Reduce debt\n"
        response += "• Diversify income sources\n"
        response += "• Don't panic sell investments\n"
        response += "• Keep skills updated\n\n"
        
        response += "**Market Volatility:**\n"
        response += "• Normal part of investing\n"
        response += "• S&P 500 has corrections (10% drops) every 1-2 years\n"
        response += "• Bear markets (20% drops) every 3-5 years\n"
        response += "• Long-term trend is up despite volatility\n"
        response += "• Stay invested, don't try to time the market\n\n"
        
        response += "**How Economic Factors Affect You:**\n\n"
        response += "• **High inflation**: Your money buys less, invest to keep pace\n"
        response += "• **Rising rates**: Lock in fixed-rate loans, savings earn more\n"
        response += "• **Falling rates**: Refinance debt, stocks may rise\n"
        response += "• **Recession**: Job security matters, keep emergency fund\n"
        response += "• **Bull market**: Stay disciplined, don't get overconfident\n"
        response += "• **Bear market**: Opportunity to buy, don't panic\n\n"
        
        response += "💡 Focus on what you can control: Savings rate, spending, career, skills.\n"
        response += "You can't control the economy, but you can prepare for any scenario!\n"
        
        return response

    
    def _handle_charity(self, query: str, context: ConversationContext, sentiment: str) -> str:
        """Handle charitable giving queries"""
        response = "Let's talk about strategic giving! ❤️\n\n"
        
        response += "**Tax-Efficient Giving Strategies:**\n\n"
        
        response += "**1. Cash Donations:**\n"
        response += "• Deductible up to 60% of AGI\n"
        response += "• Must itemize to benefit\n"
        response += "• Keep receipts for amounts > $250\n"
        response += "• Use IRS-qualified charities only\n\n"
        
        response += "**2. Donate Appreciated Stock:**\n"
        response += "• Avoid capital gains tax\n"
        response += "• Deduct full market value\n"
        response += "• Must hold > 1 year\n"
        response += "• Example: Stock worth $10k (cost $5k)\n"
        response += "  - Donate stock: $10k deduction, $0 tax\n"
        response += "  - Sell & donate cash: $10k deduction, but pay tax on $5k gain\n\n"
        
        response += "**3. Donor-Advised Fund (DAF):**\n"
        response += "• Contribute now, grant to charities later\n"
        response += "• Immediate tax deduction\n"
        response += "• Investments grow tax-free\n"
        response += "• Good for lumpy income years\n"
        response += "• Platforms: Fidelity Charitable, Schwab Charitable\n\n"
        
        response += "**4. Qualified Charitable Distribution (QCD):**\n"
        response += "• Age 70½+: Donate from IRA directly to charity\n"
        response += "• Up to $100k/year\n"
        response += "• Counts toward RMD\n"
        response += "• Excluded from taxable income\n\n"
        
        response += "**5. Bunching Donations:**\n"
        response += "• Combine 2-3 years of giving into one year\n"
        response += "• Itemize in high-giving year\n"
        response += "• Take standard deduction other years\n"
        response += "• Use DAF to spread grants over time\n\n"
        
        response += "**How Much to Give:**\n\n"
        response += "• Many aim for 10% of income (tithing)\n"
        response += "• Start with what feels comfortable (even 1-2%)\n"
        response += "• Increase as income grows\n"
        response += "• Balance with your financial goals\n\n"
        
        response += "**Choosing Charities:**\n\n"
        response += "• Research effectiveness: CharityNavigator.org, GiveWell.org\n"
        response += "• Look for low overhead (< 25%)\n"
        response += "• Align with your values\n"
        response += "• Consider local vs. global impact\n"
        response += "• Verify 501(c)(3) status\n\n"
        
        response += "**Non-Cash Giving:**\n\n"
        response += "• Volunteer time (not tax-deductible, but valuable)\n"
        response += "• Donate goods (deduct fair market value)\n"
        response += "• Professional services (pro bono)\n"
        response += "• Blood donation\n\n"
        
        response += "💡 Giving is most impactful when it's sustainable and strategic!\n"
        
        return response
    
    def _handle_behavioral(self, query: str, context: ConversationContext, sentiment: str) -> str:
        """Handle behavioral finance queries"""
        response = "Let's talk about the psychology of money! 🧠\n\n"
        
        response += "**Common Behavioral Biases:**\n\n"
        
        response += "**1. Loss Aversion:**\n"
        response += "• Pain of losing > joy of gaining\n"
        response += "• Leads to: Holding losers too long, selling winners too early\n"
        response += "• Solution: Focus on long-term goals, not daily fluctuations\n\n"
        
        response += "**2. Recency Bias:**\n"
        response += "• Overweight recent events\n"
        response += "• Leads to: Buying high (after gains), selling low (after losses)\n"
        response += "• Solution: Remember markets are cyclical, stick to plan\n\n"
        
        response += "**3. Confirmation Bias:**\n"
        response += "• Seek info that confirms beliefs\n"
        response += "• Leads to: Ignoring warning signs\n"
        response += "• Solution: Actively seek opposing viewpoints\n\n"
        
        response += "**4. Herd Mentality:**\n"
        response += "• Follow the crowd\n"
        response += "• Leads to: Bubbles and crashes\n"
        response += "• Solution: Be contrarian when appropriate\n\n"
        
        response += "**5. Overconfidence:**\n"
        response += "• Overestimate abilities\n"
        response += "• Leads to: Excessive trading, concentration risk\n"
        response += "• Solution: Acknowledge you can't predict markets\n\n"
        
        response += "**6. Mental Accounting:**\n"
        response += "• Treat money differently based on source\n"
        response += "• Leads to: Spending windfalls wastefully\n"
        response += "• Solution: Money is money, regardless of source\n\n"
        
        response += "**Building Better Money Habits:**\n\n"
        
        response += "**Automate Everything:**\n"
        response += "• Savings transfers\n"
        response += "• Bill payments\n"
        response += "• Investment contributions\n"
        response += "• Removes emotion and forgetfulness\n\n"
        
        response += "**Create Friction for Bad Habits:**\n"
        response += "• Delete shopping apps\n"
        response += "• Unsubscribe from marketing emails\n"
        response += "• Use cash for discretionary spending\n"
        response += "• 24-hour rule for purchases > $100\n\n"
        
        response += "**Make Good Habits Easy:**\n"
        response += "• Set up automatic transfers on payday\n"
        response += "• Use apps that round up purchases to savings\n"
        response += "• Make investing the default (auto-increase 401k)\n\n"
        
        response += "**Emotional Spending Triggers:**\n\n"
        response += "• Stress → Retail therapy\n"
        response += "• Boredom → Online shopping\n"
        response += "• Social pressure → Keeping up with others\n"
        response += "• Celebration → Overspending\n\n"
        
        response += "**Healthier Alternatives:**\n"
        response += "• Stress: Exercise, meditation, talk to friend\n"
        response += "• Boredom: Free activities, hobbies, learning\n"
        response += "• Social: Suggest budget-friendly activities\n"
        response += "• Celebration: Enjoy without overspending\n\n"
        
        response += "**The Best Investment Strategy:**\n"
        response += "1. Start early\n"
        response += "2. Invest regularly\n"
        response += "3. Diversify broadly\n"
        response += "4. Keep costs low\n"
        response += "5. Stay the course\n"
        response += "6. Ignore the noise\n\n"
        
        response += "💡 Your behavior matters more than market returns!\n"
        response += "A mediocre plan executed consistently beats a perfect plan abandoned.\n"
        
        return response
    
    def _handle_career(self, query: str, context: ConversationContext, sentiment: str) -> str:
        """Handle career and income optimization queries"""
        response = "Let's optimize your income! 💼\n\n"
        
        response += "**Salary Negotiation:**\n\n"
        
        response += "**Research Phase:**\n"
        response += "• Use Glassdoor, Payscale, Levels.fyi\n"
        response += "• Know your market value\n"
        response += "• Consider total compensation (benefits, bonus, equity)\n"
        response += "• Factor in location and cost of living\n\n"
        
        response += "**Negotiation Tips:**\n"
        response += "• Always negotiate (even if uncomfortable)\n"
        response += "• Let them make first offer\n"
        response += "• Ask for 10-20% more than target\n"
        response += "• Focus on value you bring\n"
        response += "• Be prepared to walk away\n"
        response += "• Get everything in writing\n\n"
        
        response += "**What to Negotiate:**\n"
        response += "• Base salary (most important)\n"
        response += "• Signing bonus\n"
        response += "• Stock options/RSUs\n"
        response += "• Performance bonus\n"
        response += "• Vacation days\n"
        response += "• Remote work flexibility\n"
        response += "• Professional development budget\n"
        response += "• Start date (to finish current projects)\n\n"
        
        response += "**Asking for a Raise:**\n\n"
        response += "**Preparation:**\n"
        response += "• Document accomplishments\n"
        response += "• Quantify impact (revenue, savings, efficiency)\n"
        response += "• Research market rates\n"
        response += "• Time it right (after wins, during reviews)\n\n"
        
        response += "**The Ask:**\n"
        response += "• Schedule dedicated meeting\n"
        response += "• Present your case professionally\n"
        response += "• Specify desired amount (10-20% typical)\n"
        response += "• Be confident but not demanding\n"
        response += "• If denied, ask what's needed to get there\n\n"
        
        response += "**Side Hustles & Additional Income:**\n\n"
        
        response += "**Freelancing:**\n"
        response += "• Use existing skills\n"
        response += "• Platforms: Upwork, Fiverr, Toptal\n"
        response += "• Set aside 25-30% for taxes\n"
        response += "• Can grow into full business\n\n"
        
        response += "**Passive Income Ideas:**\n"
        response += "• Rental property\n"
        response += "• Dividend stocks\n"
        response += "• Create digital products (courses, ebooks)\n"
        response += "• Affiliate marketing\n"
        response += "• Rent out space (Airbnb)\n"
        response += "• Rent out car (Turo)\n\n"
        
        response += "**Career Change Financial Planning:**\n\n"
        response += "**Before You Quit:**\n"
        response += "• Save 6-12 months expenses\n"
        response += "• Pay down debt\n"
        response += "• Research new field thoroughly\n"
        response += "• Build skills while employed\n"
        response += "• Network in target industry\n"
        response += "• Have health insurance plan\n\n"
        
        response += "**Income Growth Strategy:**\n\n"
        response += "1. **Maximize current role**: Raises, promotions\n"
        response += "2. **Job hop strategically**: 10-20% raises\n"
        response += "3. **Develop high-value skills**: Tech, leadership, sales\n"
        response += "4. **Side income**: Freelance, consulting\n"
        response += "5. **Invest in yourself**: Education, certifications\n"
        response += "6. **Build network**: Opportunities come from connections\n\n"
        
        if context.user_profile['income']:
            current = context.user_profile['income'] * 12
            response += f"\n**Your Income Growth Potential:**\n"
            response += f"Current: ${current:,.0f}/year\n"
            response += f"With 10% raise: ${current * 1.10:,.0f}/year (+${current * 0.10:,.0f})\n"
            response += f"With 20% raise: ${current * 1.20:,.0f}/year (+${current * 0.20:,.0f})\n"
            response += f"With $500/month side hustle: ${current + 6000:,.0f}/year\n"
        
        response += "\n💡 Your income is your greatest wealth-building tool. Invest in growing it!\n"
        
        return response

    
    def _handle_general(self, query: str, context: ConversationContext, sentiment: str) -> str:
        """Handle general queries and fallback"""
        
        # Check for greetings
        greetings = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
        if any(greet in query.lower() for greet in greetings):
            responses = [
                "Hello! I'm here to help with your financial questions. What would you like to discuss?",
                "Hi there! Ready to tackle your financial goals together. What's on your mind?",
                "Hey! I'm your local financial advisor. How can I assist you today?",
            ]
            return random.choice(responses)
        
        # Check for thanks
        thanks = ['thank', 'thanks', 'appreciate']
        if any(word in query.lower() for word in thanks):
            responses = [
                "You're welcome! Feel free to ask anything else about your finances.",
                "Happy to help! Let me know if you have more questions.",
                "My pleasure! I'm here whenever you need financial guidance.",
            ]
            return random.choice(responses)
        
        # General financial advice
        response = "I'm here to help with all aspects of your financial life! 💰\n\n"
        response += "**I can assist with:**\n\n"
        response += "• 📊 Budgeting and expense tracking\n"
        response += "• 💰 Savings strategies and emergency funds\n"
        response += "• 💳 Debt management and payoff plans\n"
        response += "• 📈 Investing and portfolio allocation\n"
        response += "• 🏖️ Retirement planning\n"
        response += "• 💼 Tax optimization strategies\n"
        response += "• 🛡️ Insurance and risk management\n"
        response += "• 📜 Estate planning basics\n"
        response += "• 💳 Credit score improvement\n"
        response += "• 🏠 Real estate and mortgages\n"
        response += "• 🎓 Education funding (529 plans)\n"
        response += "• 💼 Business finance\n"
        response += "• ⚠️ Alternative investments\n"
        response += "• 🎯 Financial goal setting\n"
        response += "• 📊 Economic factors and inflation\n"
        response += "• ❤️ Charitable giving strategies\n"
        response += "• 🧠 Behavioral finance and money psychology\n"
        response += "• 💼 Career and income optimization\n\n"
        
        response += "**Quick Tips:**\n"
        response += "• Share your numbers (income, expenses, age) for personalized advice\n"
        response += "• Ask specific questions for detailed guidance\n"
        response += "• All conversations are private and local\n\n"
        
        response += "What would you like to explore?\n"
        
        return response


# ============================================================================
# MAIN FINAI CLASS
# ============================================================================

class FinAI:
    """Main FinAI application"""
    
    def __init__(self):
        self.model = FinAIModel()
        self.nlp = NLPProcessor()
        self.calc = FinancialCalculators()
        self.context = ConversationContext()
        self.response_gen = ResponseGenerator(self.model, self.nlp, self.calc)
    
    def initialize(self):
        """Initialize the AI system"""
        print("=" * 70)
        print("🤖 FinAI - Your Local Financial Advisor")
        print("=" * 70)
        print()
        
        # Initialize NLTK
        initialize_nltk()
        
        # Try to load existing model
        print("Loading AI model...")
        if self.model.load():
            print("✓ Loaded pre-trained model from disk\n")
        else:
            print("No existing model found. Training new model...")
            self.model.train(Config.TRAINING_SAMPLES)
            self.model.save()
            print("✓ Model saved for future use\n")
        
        # Welcome message
        print("Welcome! I'm FinAI, your comprehensive local financial advisor.")
        print("I've been trained on extensive synthetic financial data to provide")
        print("intelligent, personalized guidance across all aspects of personal finance.")
        print()
        print("💡 Key Features:")
        print("   • 100% local - no internet required, complete privacy")
        print("   • Custom-trained AI model for intent classification")
        print("   • Advanced financial calculations and simulations")
        print("   • Context-aware conversations")
        print("   • Covers 18+ financial topics comprehensively")
        print()
        print("⚠️  Disclaimer: This is general information, not personalized financial advice.")
        print("   Consult a licensed professional for your specific situation.")
        print()
        print("Type 'exit', 'quit', or 'bye' to end the conversation.")
        print("=" * 70)
        print()
    
    def chat(self):
        """Main chat loop"""
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                    print("\nFinAI: Thank you for using FinAI! Remember:")
                    print("   • Pay yourself first")
                    print("   • Invest for the long term")
                    print("   • Live below your means")
                    print("   • Keep learning about finance")
                    print("\nGood luck on your financial journey! 💰\n")
                    break
                
                # Add to context
                self.context.add_message('user', user_input)
                
                # Generate response
                response = self.response_gen.generate_response(user_input, self.context)
                
                # Add to context
                self.context.add_message('assistant', response)
                
                # Display response
                print(f"\nFinAI: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nFinAI: Goodbye! Stay financially savvy! 💰\n")
                break
            except Exception as e:
                print(f"\nFinAI: I encountered an error: {str(e)}")
                print("Let's try that again. What would you like to know?\n")
    
    def run(self):
        """Run the FinAI application"""
        self.initialize()
        self.chat()


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    try:
        finai = FinAI()
        finai.run()
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        print("Please ensure all required packages are installed:")
        print("  pip install numpy pandas scikit-learn nltk")
        print("  pip install numpy-financial  # optional but recommended")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
