import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from record_news import parsear, validar  # noqa: E402

RSS = b"""<?xml version="1.0"?><rss><channel>
<item><title>BTC &amp; ETF</title><link>https://a/1</link><pubDate>Fri, 17 Oct 2026</pubDate>
<description>&lt;p&gt;Hola  mundo&lt;/p&gt;</description></item>
<item><title></title><link>https://a/2</link></item>
</channel></rss>"""

ATOM = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Fed sube tipos</title>
<link href="https://fed/1"/><updated>2026-10-17</updated><summary>texto</summary></entry></feed>"""


def test_parsear_rss_y_atom():
    assert parsear(RSS) == [('BTC & ETF', 'https://a/1', 'Fri, 17 Oct 2026', 'Hola mundo')]
    assert parsear(ATOM) == [('Fed sube tipos', 'https://fed/1', '2026-10-17', 'texto')]


def test_validar_fuerza_rangos_y_rechaza_basura():
    v = validar('{"relevante": true, "sesgo": 7, "riesgo": -1, "tema": "luna"}')
    assert v == {'relevante': True, 'sesgo': 2, 'riesgo': 0, 'tema': 'otro'}
    assert validar('{"sesgo": 1.5, "riesgo": 1.2}')['sesgo'] == 2
    assert validar('no es json') is None
    assert validar('{"riesgo": 1}') is None
