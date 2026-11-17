"""Real browser ONNX inference, parity, language switching and mobile overflow."""
import json,sys
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
url=sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:8788'
out=ROOT/'docs/screenshots';out.mkdir(exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
    errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto(url,wait_until='networkidle');page.locator('#run:not([disabled])').wait_for(timeout=30000)
    assert page.locator('h1').inner_text().startswith('Does more context')
    page.screenshot(path=str(out/'desktop-input.png'),full_page=False)
    inventory=page.evaluate('window.__manifest')
    reference={row['id']:row['expected_probabilities'] for row in json.loads((ROOT/'docs/export-parity.json').read_text())['cases']}
    results=[]
    # All displayed cases include errors: no selection on model agreement.
    for i,case in enumerate(inventory['cases']):
        page.select_option('#case-select',str(i));page.evaluate('window.__lastInference=null')
        page.click('#run');page.wait_for_function('window.__lastInference !== null',timeout=120000)
        result=page.evaluate('window.__lastInference');assert result['case_id']==case['id']
        error=max(abs(v-reference[case['id']][name][j]) for name,values in result['scores'].items() for j,v in enumerate(values))
        assert error<.003,(case['id'],error)
        results.append({'case_id':case['id'],'max_probability_error':error,'elapsed_ms':result['elapsed_ms']})
    page.locator('#prediction').scroll_into_view_if_needed();page.screenshot(path=str(out/'desktop-inference.png'),full_page=False)
    page.click('#language');assert page.locator('h1').inner_text().startswith('除了照片')
    assert page.locator('#prediction-cards h4').first.inner_text()=='只看照片'
    page.locator('h1').scroll_into_view_if_needed();page.screenshot(path=str(out/'chinese-input.png'),full_page=False)
    page.set_viewport_size({'width':390,'height':844});page.evaluate('window.scrollTo(0,0)')
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'mobile overflow'
    page.screenshot(path=str(out/'mobile.png'),full_page=True)
    assert not errors,errors
    report={'url':url,'cases_tested':len(results),'real_inference':True,'browser':browser.version,'language_switch':'passed','mobile_overflow':False,'console_errors':errors,'cases':results}
    (ROOT/'docs/browser-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'cases_tested':len(results),'maximum_error':max(v['max_probability_error'] for v in results),'language':'pass','mobile':'pass','page_errors':errors}))
    browser.close()
