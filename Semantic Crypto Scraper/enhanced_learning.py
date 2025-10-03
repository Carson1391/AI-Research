from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, deque
import numpy as np
import json
import time
import logging
from storage import DataStorage
from memory_integration import IntegratedMemory, MemoryConfig

@dataclass
class LearningPattern:
    """Enhanced pattern tracking with market focus"""
    pattern_type: str
    pattern_data: Dict[str, Any]
    success_count: int = 0
    fail_count: int = 0
    confidence: float = 0.5
    last_updated: float = field(default_factory=time.time)
    relevance_score: float = 0.0
    market_impact: float = 0.0  # Added market impact tracking
    volume_correlation: float = 0.0  # Added volume correlation
    price_trends: List[float] = field(default_factory=list)  # Track price trends

@dataclass
class ContentPriority:
    """Enhanced content priority scoring"""
    crypto_relevance: float = 1.0
    technical_value: float = 0.8
    market_relevance: float = 0.9
    trading_signals: float = 0.9  # Added trading signals priority
    pattern_strength: float = 0.8  # Added pattern strength
    volume_profile: float = 0.7   # Added volume profile
    general_value: float = 0.3
    
    def get_priority_score(self, content_type: str, data: Dict[str, Any] = None) -> float:
        base_score = {
            'crypto': self.crypto_relevance,
            'technical': self.technical_value,
            'market': self.market_relevance,
            'trading': self.trading_signals,
            'pattern': self.pattern_strength,
            'volume': self.volume_profile,
            'general': self.general_value
        }.get(content_type, 0.1)
        
        # Apply data-specific modifiers
        if data and 'price_change' in data:
            if abs(data['price_change']) > 5:  # Significant price change
                base_score *= 1.2
                
        if data and 'volume_surge' in data:
            if data['volume_surge'] > 2:  # Volume more than 2x average
                base_score *= 1.3
                
        return min(base_score, 1.0)  # Cap at 1.0

@dataclass
class RewardConfig:
    """Enhanced reward system configuration"""
    # Market data rewards
    market_rewards: Dict[str, int] = field(default_factory=lambda: {
        'price_trend_identified': 100,
        'volume_pattern_found': 80,
        'support_resistance_found': 90,
        'breakout_detected': 120,
        'correlation_found': 70
    })
    
    # Technical analysis rewards
    technical_rewards: Dict[str, int] = field(default_factory=lambda: {
        'pattern_confirmed': 100,
        'indicator_signal': 80,
        'trend_validation': 90,
        'divergence_found': 100,
        'momentum_shift': 85
    })
    
    # Dataset quality rewards
    dataset_rewards: Dict[str, int] = field(default_factory=lambda: {
        'high_quality': 100,
        'proper_format': 50,
        'comprehensive': 75,
        'consistent': 60,
        'relevant_content': 200
    })
    
    # Pattern recognition rewards
    pattern_rewards: Dict[str, int] = field(default_factory=lambda: {
        'new_pattern_identified': 150,
        'pattern_validated': 100,
        'false_pattern_avoided': 80,
        'pattern_combination': 120
    })
    
    def get_reward(self, category: str, action: str) -> int:
        """Get reward value with category-specific bonuses"""
        reward_dict = getattr(self, f"{category}_rewards", {})
        base_reward = reward_dict.get(action, 0)
        
        # Apply category-specific bonuses
        if category == 'market' and action.startswith('breakout'):
            base_reward *= 1.5  # Breakouts are important
        elif category == 'pattern' and action.startswith('new_pattern'):
            base_reward *= 1.2  # Encourage pattern discovery
            
        return int(base_reward)

class EnhancedLearning:
    def __init__(self, data_interface: DataStorage, learning_memory: Optional[IntegratedMemory] = None):
        """Initialize learning system with data interface and IntegratedMemory"""
        self.data = data_interface
        # Assign the provided learning memory instance or create a new one with configuration
        self.learning_memory = learning_memory or IntegratedMemory(MemoryConfig())
        self.logger = logging.getLogger("EnhancedLearning")

        # Initialize components
        self.priorities = ContentPriority()  # Assuming ContentPriority is correctly imported or defined
        self.rewards = RewardConfig()  # Assuming RewardConfig is correctly imported or defined

        # Enhanced pattern tracking
        self.active_patterns: Dict[str, LearningPattern] = {}
        self.pattern_scores: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        
        # Market state tracking
        self.market_state = {
            'trends': defaultdict(list),
            'volumes': defaultdict(list),
            'correlations': defaultdict(float),
            'support_levels': defaultdict(list),
            'resistance_levels': defaultdict(list)
        }
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup enhanced logging"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def process_market_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data with enhanced pattern detection"""
        try:
            symbol = data.get('symbol')
            timestamp = data.get('timestamp', time.time())
            
            # Update market state
            if 'price' in data:
                self.market_state['trends'][symbol].append(
                    (timestamp, data['price'])
                )
            if 'volume' in data:
                self.market_state['volumes'][symbol].append(
                    (timestamp, data['volume'])
                )
            
            results = {'patterns': [], 'signals': []}
            
            # Detect patterns
            patterns = self._detect_patterns(symbol)
            if patterns:
                results['patterns'].extend(patterns)
                self.logger.info(f"Detected patterns: {patterns}")
            
            # Generate signals
            signals = self._generate_signals(symbol)
            if signals:
                results['signals'].extend(signals)
                self.logger.info(f"Generated signals: {signals}")
                
            # Store in IntegratedMemory if significant
            if patterns or signals:
                self.IntegratedMemory.remember(
                    IntegratedMemory.GROWTH_PATTERN,
                    {
                        'symbol': symbol,
                        'patterns': patterns,
                        'signals': signals,
                        'timestamp': timestamp
                    }
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            return {}

    def _detect_patterns(self, symbol: str) -> List[Dict[str, Any]]:
        """Enhanced pattern detection with market context"""
        patterns = []
        try:
            prices = self.market_state['trends'].get(symbol, [])
            volumes = self.market_state['volumes'].get(symbol, [])
            
            if len(prices) < 10:  # Need enough data
                return patterns
                
            # Get recent price data
            recent_prices = [p[1] for p in prices[-10:]]
            recent_volumes = [v[1] for v in volumes[-10:]]
            
            # Calculate key metrics
            price_change = (recent_prices[-1] / recent_prices[0] - 1) * 100
            volume_change = (recent_volumes[-1] / np.mean(recent_volumes[:-1]) - 1) * 100
            price_volatility = np.std(recent_prices) / np.mean(recent_prices)
            
            # Pattern detection logic
            if self._is_breakout_pattern(recent_prices, recent_volumes):
                patterns.append({
                    'type': 'breakout',
                    'confidence': 0.8,
                    'price_change': price_change,
                    'volume_surge': volume_change
                })
                
            if self._is_accumulation_pattern(recent_prices, recent_volumes):
                patterns.append({
                    'type': 'accumulation',
                    'confidence': 0.7,
                    'duration': len(recent_prices),
                    'volume_profile': 'increasing'
                })
                
            if price_change > 5 and volume_change > 50:
                patterns.append({
                    'type': 'volume_breakout',
                    'confidence': 0.75,
                    'price_change': price_change,
                    'volume_change': volume_change
                })
                
            # Store patterns for learning
            for pattern in patterns:
                pattern_id = f"{symbol}_{pattern['type']}_{time.time()}"
                self.active_patterns[pattern_id] = LearningPattern(
                    pattern_type=pattern['type'],
                    pattern_data={
                        'symbol': symbol,
                        'metrics': {
                            'price_change': price_change,
                            'volume_change': volume_change,
                            'volatility': price_volatility
                        }
                    }
                )
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {e}")
            return []

    def _is_breakout_pattern(self, prices: List[float], volumes: List[float]) -> bool:
        """Detect breakout patterns with volume confirmation"""
        try:
            if len(prices) < 5 or len(volumes) < 5:
                return False
                
            # Calculate moving averages
            ma5 = np.mean(prices[-5:])
            ma10 = np.mean(prices[-10:]) if len(prices) >= 10 else ma5
            
            # Check for price breakout
            price_breakout = prices[-1] > max(prices[:-1])
            
            # Check for volume confirmation
            recent_volume_avg = np.mean(volumes[-3:])
            prior_volume_avg = np.mean(volumes[:-3])
            volume_surge = recent_volume_avg > prior_volume_avg * 1.5
            
            return price_breakout and volume_surge and (ma5 > ma10)
            
        except Exception as e:
            self.logger.error(f"Error in breakout detection: {e}")
            return False

    def _is_accumulation_pattern(self, prices: List[float], volumes: List[float]) -> bool:
        """Detect accumulation patterns with volume analysis"""
        try:
            if len(prices) < 7 or len(volumes) < 7:
                return False
                
            # Price is trading sideways
            price_range = max(prices) - min(prices)
            avg_price = np.mean(prices)
            price_volatility = price_range / avg_price
            
            # Volume is increasing
            volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
            
            return price_volatility < 0.03 and volume_trend > 0
            
        except Exception as e:
            self.logger.error(f"Error in accumulation detection: {e}")
            return False

    def _generate_signals(self, symbol: str) -> List[Dict[str, Any]]:
        """Generate trading signals with confidence levels"""
        signals = []
        try:
            prices = self.market_state['trends'].get(symbol, [])
            volumes = self.market_state['volumes'].get(symbol, [])
            
            if len(prices) < 20:  # Need enough historical data
                return signals
                
            # Get price and volume data
            price_data = np.array([p[1] for p in prices[-20:]])
            volume_data = np.array([v[1] for v in volumes[-20:]])
            
            # Calculate technical indicators
            ma7 = np.mean(price_data[-7:])
            ma20 = np.mean(price_data)
            volume_ma5 = np.mean(volume_data[-5:])
            volume_ma20 = np.mean(volume_data)
            
            current_price = price_data[-1]
            price_change = (current_price / price_data[-2] - 1) * 100
            
            # Trend analysis
            trend = self._analyze_trend(price_data)
            volume_trend = self._analyze_trend(volume_data)
            
            # Generate signals based on conditions
            if trend['direction'] == 'up' and volume_trend['direction'] == 'up':
                if current_price > ma7 and ma7 > ma20:
                    signals.append({
                        'type': 'strong_uptrend',
                        'confidence': 0.85,
                        'price_change': price_change,
                        'volume_confirmation': True,
                        'indicators': {
                            'ma7': ma7,
                            'ma20': ma20,
                            'volume_ratio': volume_ma5 / volume_ma20
                        }
                    })
                    
            elif trend['direction'] == 'down' and price_data[-1] < ma7 < ma20:
                signals.append({
                    'type': 'downtrend',
                    'confidence': 0.75,
                    'price_change': price_change,
                    'volume_confirmation': volume_trend['direction'] == 'up',
                    'indicators': {
                        'ma7': ma7,
                        'ma20': ma20,
                        'volume_ratio': volume_ma5 / volume_ma20
                    }
                })
            
            # Check for reversal signals
            reversal = self._check_reversal(price_data, volume_data)
            if reversal:
                signals.append(reversal)
            
            # Store signals in IntegratedMemory if significant
            if signals:
                self.IntegratedMemory.remember(
                    IntegratedMemory.INSIGHT,
                    {
                        'symbol': symbol,
                        'signals': signals,
                        'timestamp': time.time(),
                        'context': {
                            'trend': trend,
                            'volume_trend': volume_trend,
                            'technical_levels': self._get_technical_levels(price_data)
                        }
                    }
                )
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return []

    def _analyze_trend(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze trend direction and strength"""
        try:
            # Calculate linear regression
            x = np.arange(len(data))
            slope, intercept = np.polyfit(x, data, 1)
            
            # Calculate R-squared
            y_pred = slope * x + intercept
            r_squared = 1 - np.sum((data - y_pred) ** 2) / np.sum((data - np.mean(data)) ** 2)
            
            return {
                'direction': 'up' if slope > 0 else 'down',
                'slope': slope,
                'strength': r_squared,
                'validated': r_squared > 0.7
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend: {e}")
            return {'direction': 'undefined', 'strength': 0}

    def _check_reversal(self, prices: np.ndarray, volumes: np.ndarray) -> Optional[Dict[str, Any]]:
        """Check for potential trend reversals"""
        try:
            if len(prices) < 10:
                return None
                
            # Get recent price action
            recent_prices = prices[-5:]
            recent_volumes = volumes[-5:]
            
            # Calculate price change
            price_change = (recent_prices[-1] / recent_prices[0] - 1) * 100
            
            # Check for reversal conditions
            if price_change > 5 and np.all(recent_volumes > np.mean(volumes)):
                # Potential bullish reversal
                if np.all(prices[-3:] > prices[-4]):
                    return {
                        'type': 'bullish_reversal',
                        'confidence': 0.8,
                        'price_change': price_change,
                        'volume_surge': np.mean(recent_volumes) / np.mean(volumes),
                        'confirmation_count': 3
                    }
                    
            elif price_change < -5 and np.all(recent_volumes > np.mean(volumes)):
                # Potential bearish reversal
                if np.all(prices[-3:] < prices[-4]):
                    return {
                        'type': 'bearish_reversal',
                        'confidence': 0.8,
                        'price_change': price_change,
                        'volume_surge': np.mean(recent_volumes) / np.mean(volumes),
                        'confirmation_count': 3
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking reversal: {e}")
            return None

    def _get_technical_levels(self, prices: np.ndarray) -> Dict[str, List[float]]:
        """Calculate key technical levels"""
        try:
            levels = {
                'support': [],
                'resistance': [],
                'pivots': []
            }
            
            if len(prices) < 20:
                return levels
                
            # Find support/resistance using rolling min/max
            window = 10
            for i in range(window, len(prices) - window):
                price_window = prices[i-window:i+window]
                current_price = prices[i]
                
                # Support level
                if current_price == np.min(price_window):
                    levels['support'].append(current_price)
                    
                # Resistance level
                if current_price == np.max(price_window):
                    levels['resistance'].append(current_price)
                    
            # Keep only significant levels (reduce noise)
            levels['support'] = self._cluster_levels(levels['support'])
            levels['resistance'] = self._cluster_levels(levels['resistance'])
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error getting technical levels: {e}")
            return {'support': [], 'resistance': [], 'pivots': []}

    def _cluster_levels(self, levels: List[float], tolerance: float = 0.02) -> List[float]:
        """Cluster nearby price levels to reduce noise"""
        if not levels:
            return []
            
        # Sort levels
        sorted_levels = sorted(levels)
        clustered = []
        current_cluster = [sorted_levels[0]]
        
        for level in sorted_levels[1:]:
            # Check if level belongs to current cluster
            if (level - current_cluster[0]) / current_cluster[0] <= tolerance:
                current_cluster.append(level)
            else:
                # Add average of current cluster
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
                
        # Add final cluster
        if current_cluster:
            clustered.append(np.mean(current_cluster))
            
        return clustered

    def process_content(self, content: Dict[str, Any], content_type: str) -> Dict[str, Any]:
        """Process content with enhanced market focus"""
        try:
            # Get current priority weights
            market_priority = self.priorities.get_priority_score('market', content)
            technical_priority = self.priorities.get_priority_score('technical', content)
            crypto_priority = self.priorities.get_priority_score('crypto', content)
            
            # Calculate individual scores
            market_score = self._evaluate_market_data(content) * market_priority
            technical_score = self._evaluate_technical_data(content) * technical_priority
            relevance_score = self._evaluate_crypto_relevance(content) * crypto_priority
            
            # Combined score with weighted importance
            total_score = (market_score * 0.4 + 
                         technical_score * 0.35 + 
                         relevance_score * 0.25)
            
            result = {
                'total_score': total_score,
                'components': {
                    'market_score': market_score,
                    'technical_score': technical_score,
                    'relevance_score': relevance_score
                },
                'priorities': {
                    'market': market_priority,
                    'technical': technical_priority,
                    'crypto': crypto_priority
                },
                'timestamp': time.time()
            }
            
            # Store insights if score is high enough
            if total_score > 0.7:
                self.IntegratedMemory.remember(
                    IntegratedMemory.INSIGHT,
                    {
                        'content_type': content_type,
                        'scores': result['components'],
                        'total_score': total_score,
                        'timestamp': time.time()
                    }
                )
            
            # Update performance history
            self.performance_history.append(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing content: {e}")
            return {}

    def _evaluate_market_data(self, content: Dict[str, Any]) -> float:
        """Evaluate market data quality and significance"""
        try:
            score = 0
            r = self.rewards.market_rewards
            
            # Check for price data
            if 'price' in content and isinstance(content['price'], (int, float)):
                score += r['price_trend_identified']
            
            # Check for volume data
            if 'volume' in content and isinstance(content['volume'], (int, float)):
                score += r['volume_pattern_found']
            
            # Check for support/resistance levels
            if 'levels' in content and content['levels']:
                score += r['support_resistance_found']
            
            # Check for breakout signals
            if 'breakout' in content and content['breakout']:
                score += r['breakout_detected']
            
            # Normalize score
            max_possible = sum(r.values())
            return score / max_possible if max_possible > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error evaluating market data: {e}")
            return 0.0

    def _evaluate_technical_data(self, content: Dict[str, Any]) -> float:
        """Evaluate technical analysis quality"""
        try:
            score = 0
            r = self.rewards.technical_rewards
            
            # Check for confirmed patterns
            if 'patterns' in content and content['patterns']:
                score += r['pattern_confirmed']
            
            # Check for indicator signals
            if 'indicators' in content and content['indicators']:
                score += r['indicator_signal']
            
            # Check for trend validation
            if 'trend' in content and content['trend'].get('validated', False):
                score += r['trend_validation']
            
            # Check for divergences
            if 'divergence' in content and content['divergence']:
                score += r['divergence_found']
            
            # Normalize score
            max_possible = sum(r.values())
            return score / max_possible if max_possible > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error evaluating technical data: {e}")
            return 0.0

    def _evaluate_crypto_relevance(self, content: Dict[str, Any]) -> float:
        """Evaluate crypto-specific relevance"""
        try:
            crypto_keywords = {
                'bitcoin', 'btc', 'ethereum', 'eth', 'blockchain',
                'crypto', 'token', 'defi', 'nft', 'mining',
                'wallet', 'exchange', 'trading', 'altcoin'
            }
            
            # Convert content to searchable text
            content_text = ' '.join(str(v).lower() for v in content.values() 
                                  if isinstance(v, (str, int, float)))
            
            # Count keyword matches
            matches = sum(1 for word in crypto_keywords 
                        if word in content_text)
            
            # Calculate relevance score
            return min(matches / 5, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.error(f"Error evaluating crypto relevance: {e}")
            return 0.0