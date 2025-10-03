import asyncio
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import base64
from vision_process import (
    extract_vision_info,
    fetch_image,
    fetch_video,
    process_vision_info,
    smart_resize,
)
from playwright.async_api import async_playwright, Browser, Page
from storage import DataStorage
from memory_integration import IntegratedMemory
from enhanced_learning import EnhancedLearning

@dataclass
class ScraperConfig:
    """Configuration for the CryptoScraper."""
    # List of URLs to scrape, organized by category
    sources: Dict[str, List[str]] = field(default_factory=lambda: {
        'charts': [
            'https://www.tradingview.com/crypto-screener/',
            'https://www.coingecko.com/en',
            'https://coinmarketcap.com',
            'https://www.binance.com/en/markets',
            'https://pro.coinbase.com/markets'
        ],
        'news': [
            'https://cointelegraph.com',
            'https://decrypt.co',
            'https://coindesk.com',
            'https://cryptonews.com'
        ],
        'sentiment': [
            'https://www.reddit.com/r/cryptocurrency/',
            'https://www.reddit.com/r/Bitcoin/',
            'https://www.reddit.com/r/Altcoin/'
        ]
    })
    # Delay range between actions to mimic human behavior
    delay_range: Tuple[float, float] = (1.0, 3.0)
    # Directory path for data storage
    data_dir: str = "data"
    # Proportions for data types in each layer
    layer_proportions: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        'early': {'abstract': 20, 'concrete': 40, 'graphs': 40},
        'middle': {'abstract': 40, 'concrete': 30, 'graphs': 30},
        'later': {'abstract': 60, 'concrete': 20, 'graphs': 20}
    })
    # Batch sizes for each layer
    layer_batch_sizes: Dict[str, int] = field(default_factory=lambda: {
        'early': 5,
        'middle': 3,
        'later': 2
    })
    # Time to spend on each website (in seconds)
    browsing_time: int = 20  # Approximately 20 seconds per page

class CryptoScraper:
    def __init__(self, config: ScraperConfig, storage: DataStorage, qwen_manager, memory=None, learning=None):
        self.config = config
        self.storage = storage
        self.qwen = qwen_manager
        self.memory = memory or self.qwen.get_memory()
        self.learning = learning or self.qwen.get_learning()
        
        # Configure logger
        self.logger = logging.getLogger("CryptoScraper")
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Initialize these as None
        self.browser = None
        self.playwright = None
        self.context = None
        
        self.stats = {
            'total_scrapes': 0,
            'successful_scrapes': 0,
            'failed_scrapes': 0
        }

    async def start_browser(self):
        """Start the Playwright browser instance."""
        try:
            self.logger.info("Starting browser - please wait...")
            
            # Ensure any existing browser instances are properly closed
            await self.close_browser()
            
            self.playwright = await async_playwright().start()
            self.logger.info("Playwright started")
            
            # Create user data directory if it doesn't exist
            user_data_dir = Path("./browser_data").absolute()
            user_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Launch browser with improved configuration
            self.browser = await self.playwright.chromium.launch(
                headless=False,
                args=[
                    '--disable-background-timer-throttling',
                    '--disable-backgrounding-occluded-windows',
                    '--disable-renderer-backgrounding',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage'
                ]
            )
            
            # Create a new context with specific viewport settings
            self.context = await self.browser.new_context(
                viewport={'width': 1200, 'height': 800},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
            
            # Add focus prevention script
            await self.context.add_init_script("""
                window.focus = function() {};
                window.alert = function() {};
                window.confirm = function() { return true; };
            """)
            
            self.logger.info("Browser context created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Browser error: {e}")
            await self.close_browser()  # Ensure cleanup on error
            return False

    async def run(self):
        """Main method to start scraping all configured sources."""
        try:
            self.logger.info("Starting scraping process...")
            success = await self.start_browser()
            
            if not success:
                self.logger.error("Browser failed to start. Aborting scraping.")
                return
                
            self.logger.info("Browser started successfully")
            
            # Create data directory if it doesn't exist
            os.makedirs(self.config.data_dir, exist_ok=True)
            
            while True:
                for category, urls in self.config.sources.items():
                    for url in urls:
                        try:
                            self.logger.info(f"Scraping {url}...")
                            page = await self.context.new_page()
                            await page.goto(url, wait_until='networkidle')
                            await self.simulate_human_behavior(page, self.config.browsing_time)
                            
                            # Intelligent navigation and data extraction
                            await self.intelligent_navigation_and_extraction(page, category)
                            
                            await page.close()
                            self.stats['successful_scrapes'] += 1
                            self.logger.info(f"Finished scraping {url}")
                        except Exception as e:
                            self.logger.error(f"Error scraping {url}: {e}")
                            self.stats['failed_scrapes'] += 1
                        finally:
                            self.stats['total_scrapes'] += 1
                            # Random delay between scrapes
                            delay = random.uniform(*self.config.delay_range)
                            self.logger.info(f"Waiting {delay:.1f}s before next scrape...")
                            await asyncio.sleep(delay)
                
                self.logger.info(f"Scraping complete! Stats: {self.stats}")

        except asyncio.CancelledError:
            self.logger.info("Scraping was cancelled.")
        except Exception as e:
            self.logger.error(f"Fatal error during scraping: {str(e)}")
        finally:
            await self.close_browser()

    async def close_browser(self):
        """Close the browser and Playwright instance."""
        try:
            if self.context:
                await self.context.close()
                self.context = None
            if self.browser:
                await self.browser.close()
                self.browser = None
            if self.playwright:
                await self.playwright.stop()
                self.playwright = None
        except Exception as e:
            self.logger.error(f"Error closing browser: {e}")
            # Force cleanup even if errors occur
            self.context = None
            self.browser = None
            self.playwright = None
    
    async def simulate_human_behavior(self, page: Page, duration: int):
        """Simulate human-like interactions on the page for a given duration."""
        self.logger.info("Simulating human behavior...")
        end_time = time.time() + duration
        while time.time() < end_time:
            try:
                # Random mouse movements
                x = random.randint(0, 1200)  # Match viewport width
                y = random.randint(0, 800)   # Match viewport height
                await page.mouse.move(x, y, steps=random.randint(5, 15))
                await asyncio.sleep(random.uniform(0.1, 0.3))

                # Random scrolling
                scroll_amount = random.randint(-200, 200)
                await page.mouse.wheel(0, scroll_amount)
                await asyncio.sleep(random.uniform(0.2, 0.5))

                # Random clicks on interactive elements
                elements = await page.query_selector_all('a, button, input[type="submit"], .interactive')
                if elements:
                    element = random.choice(elements)
                    try:
                        await element.hover()
                        await asyncio.sleep(random.uniform(0.2, 0.5))
                        await element.click()
                        self.logger.info("Clicked on an interactive element.")
                        await asyncio.sleep(random.uniform(1.0, 2.0))
                    except Exception:
                        continue

                # Random delay to mimic reading
                await asyncio.sleep(random.uniform(1.0, 3.0))
            except Exception as e:
                self.logger.error(f"Error in simulate_human_behavior: {e}")
                break

    async def intelligent_navigation_and_extraction(self, page: Page, category: str):
        """Navigate intelligently through the page and extract relevant data."""
        extraction_config = self.learning.get_extraction_config(category)
        if not extraction_config:
            self.logger.error(f"No extraction config found for category: {category}")
            return
        
        try:
            # Example: Click on links and extract data based on config
            elements = await page.query_selector_all(extraction_config.get('link_selector', 'a'))
            for element in elements:
                try:
                    # Ensure element is attached and visible
                    if not await element.is_visible():
                        continue
                    
                    # Attempt to click with retry
                    for attempt in range(3):
                        try:
                            await element.click()
                            break
                        except Exception as e:
                            if attempt == 2:  # Last attempt
                                self.logger.warning(f"Failed to click element after 3 attempts: {e}")
                                continue
                            await asyncio.sleep(1)  # Wait before retrying
                    
                    await asyncio.sleep(random.uniform(1.0, 3.0))  # Simulate reading
                    # Extract data after navigation
                    data = await self.extract_data(page, category, 'layer', 'data_type')
                    if data:
                        self.storage.store_data(data, {'category': category})
                        self.logger.info(f"Data successfully extracted and stored for {category}.")
                except Exception as e:
                    self.logger.error(f"Error processing element: {e}")
                    continue
        except Exception as e:
            self.logger.error(f"Error in intelligent_navigation_and_extraction: {e}")

    async def scrape_page(self, url: str, category: str, layer: str, data_type: str):
        """Scrape a single page and collect data."""
        self.logger.info(f"Scraping {url} for layer {layer}, data type {data_type}...")
        page = None
        try:
            page = await self.context.new_page()
            
            # Prevent new pages from taking focus
            await page.evaluate("""
                window.open = function(url) {
                    console.log('Blocked automatic window open: ' + url);
                    return null;
                };
            """)
            
            await page.goto(url, timeout=60000, wait_until='networkidle')
            await asyncio.sleep(random.uniform(*self.config.delay_range))
            
            # Keep the page in the background
            await page.evaluate("window.focus = function() {}")
            
            # Rest of the scraping logic
            await self.simulate_human_behavior(page, self.config.browsing_time)
            data = await self.extract_data(page, category, layer, data_type)
            self.stats['successful_scrapes'] += 1
            self.logger.info(f"Finished scraping {url}")
            
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {e}")
            self.stats['failed_scrapes'] += 1
        finally:
            if page:
                try:
                    await page.close()
                except Exception:
                    pass
            self.stats['total_scrapes'] += 1

    async def extract_data(self, page: Page, category: str, layer: str, data_type: str):
        """Extract data from the page based on the category and data type."""
        self.logger.info(f"Extracting {category} data ({data_type})...")
        data = {'category': category, 'timestamp': datetime.now().isoformat(), 'data_type': data_type}
        
        try:
            # Get extraction config from learning system
            extraction_config = self.learning.get_extraction_config(category)
            if not extraction_config:
                self.logger.error(f"No extraction config found for category: {category}")
                return data
            
            # Extract based on type
            if extraction_config['type'] == 'screenshot':
                # Take full page screenshot for charts
                if data_type in ['chart', 'technical']:
                    screenshot = await page.screenshot(full_page=True)
                else:
                    screenshot = await page.screenshot()
                
                # Build analysis prompt based on data type
                if data_type == 'chart':
                    prompt = "Analyze this cryptocurrency chart in detail. Identify: 1. Chart type (candlestick, line, etc) 2. Current price trend 3. Support/resistance levels 4. Volume patterns 5. Notable technical patterns 6. Key price levels"
                elif data_type == 'technical':
                    prompt = "Perform technical analysis on this chart. Focus on: 1. Moving averages 2. RSI levels 3. MACD signals 4. Bollinger bands 5. Trading volume analysis 6. Pattern formations (head & shoulders, triangles, etc)"
                else:
                    prompt = extraction_config['prompt']
                
                # Process with Qwen vision
                conversation = [{
                    "role": "user",
                    "content": prompt,
                    "images": [{"image": screenshot}]
                }]
                processed_vision = process_vision_info(conversation)
                response = await self.qwen.chat(processed_vision)
                
                # Encode screenshot for storage
                screenshot_b64 = base64.b64encode(screenshot).decode('utf-8')
                timestamp = datetime.now().isoformat()
                
                # Prepare content for storage
                content = {
                    'analysis': response,
                    'screenshot': screenshot_b64,
                    'url': page.url,
                    'timestamp': timestamp,
                    'data_type': data_type
                }
                
                # For chart and technical analysis, store in both decision and AI datasets
                if data_type in ['chart', 'technical']:
                    # Store in decision data for trading decisions
                    decision_metadata = {
                        'source': page.url,
                        'analysis_type': data_type,
                        'timestamp': timestamp,
                        'layer': layer  # Include layer info for cross-reference
                    }
                    
                    # Charts go to trend_lines, technical goes to oscillators
                    if data_type == 'chart':
                        self.storage.store_decision_data('technical_data', 'trend_lines', content, decision_metadata)
                        self.storage.store_decision_data('market_data', 'price_data', content, decision_metadata)
                    else:  # technical
                        self.storage.store_decision_data('technical_data', 'oscillators', content, decision_metadata)
                        self.storage.store_decision_data('technical_data', 'moving_averages', content, decision_metadata)
                    
                    # Store in AI dataset for training
                    ai_metadata = {
                        'source': page.url,
                        'analysis_type': data_type,
                        'timestamp': timestamp,
                        'decision_category': 'technical_data',  # Reference back to decision data
                        'decision_subcategory': 'trend_lines' if data_type == 'chart' else 'oscillators'
                    }
                    self.storage.store_ai_data(layer, 'graphs', content, ai_metadata)
                    
                    # Also store as concrete data for pattern learning
                    self.storage.store_ai_data(layer, 'concrete', {
                        'patterns': response,
                        'source_graph': content['screenshot'],
                        'timestamp': timestamp
                    }, ai_metadata)
                    
                    # Use store_data to ensure it's properly indexed in both systems
                    self.storage.store_data(content, {
                        'category': 'technical_data',
                        'subcategory': 'trend_lines' if data_type == 'chart' else 'oscillators',
                        'layer': layer,
                        'data_type': 'graphs',
                        'source': page.url,
                        'timestamp': timestamp,
                        'is_training_data': True,
                        'is_decision_data': True
                    })
                
            elif extraction_config['type'] == 'text':
                selectors = extraction_config['selectors']
                items = []
                elements = await page.query_selector_all(selectors.get('post', 'article'))
                for element in elements[:5]:
                    content = await element.evaluate('el => el.innerText')
                    prompt = extraction_config['prompt']
                    response = await self.qwen.chat([{
                        "role": "user", 
                        "content": f"{prompt}\n\nContent:\n{content}"
                    }])
                    
                    timestamp = datetime.now().isoformat()
                    item_data = {
                        'content': content,
                        'analysis': response,
                        'url': page.url,
                        'timestamp': timestamp
                    }
                    items.append(item_data)
                    
                    # Store text content in both systems too
                    text_metadata = {
                        'source': page.url,
                        'timestamp': timestamp,
                        'layer': layer
                    }
                    
                    # Store in decision data
                    if 'news' in category.lower():
                        self.storage.store_decision_data('news_data', 'announcements', item_data, text_metadata)
                    elif 'social' in category.lower():
                        self.storage.store_decision_data('social_data', 'reddit_sentiment', item_data, text_metadata)
                    
                    # Store in AI dataset
                    self.storage.store_ai_data(layer, 'abstract', item_data, text_metadata)
                    
                    # Combined storage
                    self.storage.store_data(item_data, {
                        'category': 'news_data' if 'news' in category.lower() else 'social_data',
                        'subcategory': 'announcements' if 'news' in category.lower() else 'reddit_sentiment',
                        'layer': layer,
                        'data_type': 'abstract',
                        'source': page.url,
                        'timestamp': timestamp,
                        'is_training_data': True,
                        'is_decision_data': True
                    })
                
                data['items'] = items
            
            # Process with learning system
            processed_content = self.learning.process_content(content, category)
            patterns = self.learning.learn_pattern(processed_content, category)

            # Update memory with findings
            if processed_content['total_score'] > 0.7:
                self.memory.remember(IntegratedMemory.INSIGHT, processed_content)
            if patterns:
                self.memory.remember(IntegratedMemory.GROWTH_PATTERN, patterns)
            
            return data

        except Exception as e:
            self.logger.error(f"Error extracting data: {str(e)}")
            return data

    def calculate_batch_counts(self, layer: str) -> Dict[str, int]:
        """Calculate the number of samples for each data type in a layer batch."""
        batch_size = self.config.layer_batch_sizes[layer]
        proportions = self.config.layer_proportions[layer]
        total_percentage = sum(proportions.values())
        batch_counts = {}
        for data_type, percentage in proportions.items():
            count = max(1, round((percentage / total_percentage) * batch_size))
            batch_counts[data_type] = count
        return batch_counts

    def select_category_for_data_type(self, data_type: str) -> str:
        """Select a category based on the data type."""
        if data_type == 'graphs':
            return 'charts'
        elif data_type == 'concrete':
            return random.choice(['charts', 'news'])
        elif data_type == 'abstract':
            return random.choice(['news', 'sentiment'])
        else:
            return 'news'

    def select_random_url(self, category: str) -> str:
        """Select a random URL from the category sources."""
        return random.choice(self.config.sources[category])