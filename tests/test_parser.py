import pytest
from unittest.mock import AsyncMock, patch
from aiohttp import ClientError

from src.parser import normalize, extract_tilda_content_html, extract_content_url, NBSP


class TestNormalize:
    def test_normalize_basic(self):
        """Проверяем базовую нормализацию текста"""
        result = normalize("  Обычный   текст  ")
        assert result == "Обычный текст"

    def test_normalize_nbsp(self):
        """Проверяем замену неразрывных пробелов"""
        text = f"Текст{NBSP}с{NBSP}nbsp"
        result = normalize(text)
        assert result == "Текст с nbsp"

    def test_normalize_unicode_chars(self):
        """Проверяем удаление специальных Unicode символов"""
        text = "Текст\u200e с\u200e символами"
        result = normalize(text)
        assert result == "Текст с символами"

    def test_normalize_multiple_spaces(self):
        """Проверяем схлопывание множественных пробелов"""
        text = "Текст    с     множественными\n\n   пробелами"
        result = normalize(text)
        assert result == "Текст с множественными пробелами"

    def test_normalize_none_input(self):
        """Проверяем обработку None"""
        result = normalize(None)
        assert result == ""

    def test_normalize_empty_string(self):
        """Проверяем обработку пустой строки"""
        result = normalize("")
        assert result == ""


class TestExtractTildaContentHtml:
    @pytest.mark.asyncio
    async def test_extract_with_h1_and_text_elements(self):
        """Проверяем извлечение контента с H1 и текстовыми элементами"""
        html = '''
        <html>
            <div id="allrecords">
                <h1>Заголовок страницы</h1>
                <div data-elem-type="text">
                    <div class="tn-atom">Первый текстовый блок</div>
                </div>
                <div data-elem-type="text">
                    <div class="tn-atom">Второй текстовый блок</div>
                </div>
                <footer id="t-footer">Подвал</footer>
            </div>
        </html>
        '''
        
        result = await extract_tilda_content_html(html)
        
        expected = [
            "Заголовок страницы",
            "Первый текстовый блок",
            "Второй текстовый блок"
        ]
        assert result == expected

    @pytest.mark.asyncio
    async def test_extract_no_h1(self):
        """Проверяем поведение при отсутствии H1"""
        html = '''
        <html>
            <div id="allrecords">
                <div data-elem-type="text">
                    <div class="tn-atom">Текстовый блок</div>
                </div>
            </div>
        </html>
        '''
        
        result = await extract_tilda_content_html(html)
        assert result == []

    @pytest.mark.asyncio
    async def test_extract_h1_in_header_ignored(self):
        """Проверяем игнорирование H1 в header"""
        html = '''
        <html>
            <div id="allrecords">
                <header>
                    <h1>Заголовок в хедере</h1>
                </header>
                <h1>Основной заголовок</h1>
                <div data-elem-type="text">
                    <div class="tn-atom">Текстовый блок</div>
                </div>
            </div>
        </html>
        '''
        
        result = await extract_tilda_content_html(html)
        
        expected = [
            "Основной заголовок",
            "Текстовый блок"
        ]
        assert result == expected

    @pytest.mark.asyncio
    async def test_extract_stops_at_footer(self):
        """Проверяем остановку парсинга на footer"""
        html = '''
        <html>
            <div id="allrecords">
                <h1>Заголовок</h1>
                <div data-elem-type="text">
                    <div class="tn-atom">До подвала</div>
                </div>
                <footer id="t-footer">
                    <div data-elem-type="text">
                        <div class="tn-atom">В подвале</div>
                    </div>
                </footer>
                <div data-elem-type="text">
                    <div class="tn-atom">После подвала</div>
                </div>
            </div>
        </html>
        '''
        
        result = await extract_tilda_content_html(html)
        
        expected = [
            "Заголовок",
            "До подвала"
        ]
        assert result == expected

    @pytest.mark.asyncio
    async def test_extract_filters_policy_text(self):
        """Проверяем фильтрацию текста с политикой"""
        html = '''
        <html>
            <div id="allrecords">
                <h1>Заголовок</h1>
                <div data-elem-type="text">
                    <div class="tn-atom">Нормальный текст</div>
                </div>
                <div data-elem-type="text">
                    <div class="tn-atom">вы соглашаетесь с нашей Политикой конфиденциальности</div>
                </div>
            </div>
        </html>
        '''
        
        result = await extract_tilda_content_html(html)
        
        expected = [
            "Заголовок",
            "Нормальный текст"
        ]
        assert result == expected

    @pytest.mark.asyncio
    async def test_extract_text_without_tn_atom(self):
        """Проверяем извлечение текста без tn-atom класса"""
        html = '''
        <html>
            <div id="allrecords">
                <h1>Заголовок</h1>
                <div data-elem-type="text">Текст без tn-atom</div>
            </div>
        </html>
        '''
        
        result = await extract_tilda_content_html(html)
        
        expected = [
            "Заголовок",
            "Текст без tn-atom"
        ]
        assert result == expected
