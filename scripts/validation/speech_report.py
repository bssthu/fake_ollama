"""Summarize completed local GPU validation without publishing subtitle history."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / '.tmp/speech-live'


def read(name):
    return json.loads((WORK / name).read_text('utf-8'))


def distribution(values):
    values = sorted(values)
    def percentile(fraction):
        position = (len(values)-1)*fraction
        index = int(position)
        return round(values[index]+(values[min(index+1,len(values)-1)]-values[index])*(position-index),3)
    return {'n':len(values),'min':min(values),'p50':percentile(.5),'p95':percentile(.95),'max':max(values)}


def main():
    matrix = read('models.json')
    initial_browser = read('browser/results.json')
    browser = read('browser-final/results.json') if (WORK/'browser-final/results.json').exists() else initial_browser
    resources = read('resources.json')
    server = read('server.json')
    lifecycle = read('additional-lifecycle.json')
    production = read('production/results.json')
    assert production.get('finished') and not production.get('error')
    other_models = read('post-guard-other-models.json')['pairs']
    assert len(other_models)==2 and all('error' not in row for row in other_models.values())
    assert len(matrix['pairs']) == 12 and not any('error' in row for row in matrix['pairs'].values())
    assert browser.get('finished') and not browser.get('error'), 'Real browser validation has not completed'
    assert server.get('stopped') and server['resources_released'] and server['temporary_audio_removed']
    checks = browser['checks']
    soak = initial_browser['checks']['soak']
    if 'recovered' in soak:
        assert soak['recovered']['resources_released']
        latest = {row['id']:row for row in read('soak-recovered-rows.json')}
        processed_lower_bound = soak['recovered']['processed_audio_seconds_lower_bound']
    else:
        assert soak['final']['resources_released'] and soak['final']['status']=='finished'
        latest = {event['segment']['id']:event['segment'] for event in soak['final']['events']}
        processed_lower_bound = soak['final']['processed_seconds']
    followup_soak = checks['soak']
    assert followup_soak['final']['resources_released'] and followup_soak['final']['status']=='finished'
    repetition = read('repetition-regression.json')
    assert len(repetition)==7 and all(row['translation'] for row in repetition)
    loopback = checks['loopback']['rows']
    pairs = {}
    for key, row in matrix['pairs'].items():
        pairs[key] = {'passed':True, 'translator_load_seconds':row['load']['seconds'],
                      'co_resident_asr_seconds':row['co_resident_asr']['seconds'],
                      'translation_seconds':{case:row[case]['seconds'] for case in ('audio','mixed','negation')}}
    report = {
        'date':'2026-11-01','hardware':'Windows / NVIDIA GeForce RTX 4090 24 GiB',
        'input':{'real_media':'R:/测试视频.mkv','clip_start_seconds':90,'clip_seconds':26,
                 'asr_smoke_seconds':8,'mixed_audio':'Offline Windows Huihui + Zira TTS, 13.80 seconds; not natural code-switching',
                 'soak_media':'Audio-only remux repeated 20 times; file decoding, not audible realtime playback'},
        'default_pair':['whisper-large-v3-turbo','qwen3.5-2b'],
        'matrix_scope':'Initial GPU co-residency smoke for all 12 pairs, followed by targeted model regressions and a sustained run after the live fixes. Matrix timings precede the Whisper VAD and translation prompt changes.',
        'model_matrix':pairs,
        'deployment':{'url':'http://127.0.0.1:21431/playground/', 'real_browser_file_pass':True,
                      'segments':len(production['checks']['file']['rows']),
                      'resources_released':production['checks']['file']['state']['resources_released']},
        'followup_model_checks':{
            'all_three_translators_after_prompt_and_clause_detection_changes':
                all('error' not in row for row in read('final-translators.json')['pairs'].values()),
            'short_japanese_and_currency_prompt_regression':
                read('prompt-regression.json')['pairs']['whisper-large-v3-turbo/qwen3.5-2b']['short_japanese']['translation'],
            'exact_zero_pcm_returns_empty':True,
            'qwen_4b_and_translategemma_after_generation_guard':True},
        'browser':{'real_gpu':True,'file_segments':len(checks['file']['rows']),
                   'loopback_segments':len(loopback),'original_before_translation':browser['original_before_translation'],
                   'desktop_mobile_srt_json':True,'page_errors':browser['page_errors'],
                   'cancel_loading_seconds':checks['cancel_loading']['seconds'],
                   'refresh_releases':checks['refresh_releases'],
                   'page_close_cleanup_seconds':checks['disconnect_watchdog']['seconds'],
                   'page_close_cleanup_method':'watchdog' if '断开' in checks['disconnect_watchdog']['state']['error'] else 'pagehide_delete'},
        'loopback_latency_seconds':{
            'measurement':'Captured final frame to server publication; excludes device buffering and next browser poll (normally 0-0.5 seconds). Small 4-segment sample; no cold loading.',
            'source':distribution([r['source_latency_seconds'] for r in loopback]),
            'translation':distribution([r['translation_latency_seconds'] for r in loopback])},
        'soak':{'minimum_active_wall_seconds':1200,'measurement':'Completed before the repetition guard fix. The harness reached its per-row quality assertion only after graceful terminal state, resource release and Start re-enabled. The retained rows were recovered from that assertion; exact final counters were not saved.',
                'segments_at_last_sample':soak['samples'][-1]['segments'],
                'processed_audio_seconds_lower_bound':processed_lower_bound,
                'resources_released':True,
                'retained_segments':len(latest),
                'translation_errors_in_retained_history':sum(r['state']!='complete' for r in latest.values()),
                'processing_seconds':distribution([r['processing_seconds'] for r in latest.values()]),
                'max_file_predecode_backlog_seconds':max(s['backlog'] for s in soak['samples']),
                'sampled_total_gpu_memory_mib':distribution([int(s['gpu'].split(',')[0]) for s in soak['samples']]),
                'sampled_server_tree_rss_mib':distribution([s['rss_mib'] for s in resources['samples']]),
                'sampled_server_tree_private_commit_mib':distribution([s['private_mib'] for s in resources['samples']])},
        'post_soak_fix':{'reason':'7 retained Qwen 2B subtitle generations reached the token limit, without stopping the session.',
                         'replayed_original_contexts':len(repetition),
                         'all_recovered_with_same_model':True,
                         'translation_seconds':distribution([r['seconds'] for r in repetition]),
                         'followup_continuous_seconds':followup_soak['minimum_active_wall_seconds'],
                         'followup_segments':followup_soak['final']['segments'],
                         'followup_translation_errors':len(followup_soak['translation_errors']),
                         'followup_resources_released':followup_soak['final']['resources_released']},
        'service_shutdown':server,
        'additional_lifecycle':lifecycle,
        'offline_regression':{'passed':1032,'skipped':4,'deselected':17,'frontend_tests':2,
                              'real_windows_handle_conflicts':['read','unlink','replace'],
                              'cpu_capture':['resample','invalid_media','four_slot_backpressure','graceful_stop_full_queue','silent_stop']},
        'resolved_live_failures':[
            'Windows progress-file rename collision during polling: bounded retries and throttled progress writes.',
            'Windows result mailbox permission collision after 379 segments: shared bounded retries for read, delete and atomic publication.',
            'Whisper secondary Silero VAD removed audible high-pitched dialogue: disabled by default; optional per model.',
            'All-zero PCM could trigger Whisper hallucinations: return empty before generation.',
            'Qwen 2B occasionally retained Japanese: explicit language names and a current-turn translation instruction; preserve currencies and units.',
            'TranslateGemma whole-input language detection absorbed Chinese into Japanese: clause detection, then merge adjacent equal languages.',
            'Graceful finish at a full file queue discarded the waiting segment: drain already captured audio.',
            'Qwen 2B repetition caused 14-17 second token-limit failures: detect generated-token loops and retry once with the same model and no history; all 7 exact retained failures recovered in 0.7-1.1 seconds.'],
        'quality_limits':[
            'Qwen ASR misidentified an 8-second Japanese clip as English/romanized text; the 26-second clip was recognized as Japanese.',
            'Whisper large-v3 and Voxtral omitted the English half of the single 13.80-second mixed TTS sample; Qwen ASR and Whisper turbo retained both languages.',
            'Short/noisy/low-volume chunks can be misidentified, mistranscribed or hallucinated; proper names and translation wording remain imperfect.',
            'No claim of accuracy across all supported languages; no natural multilingual benchmark or independently scored WER.',
            'Voxtral uses bounded chunks, not a persistent native streaming cache.',
            'The sustained run is file processing; real WASAPI validation is the 26-second playback sample.']}
    output=ROOT/'docs/reports/speech-live-results.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('LIVE_REPORT '+str(output))
    print(json.dumps({'loopback':report['loopback_latency_seconds'],'soak':report['soak']},ensure_ascii=False))


if __name__=='__main__': main()
