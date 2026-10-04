const state = { selected: null, upload: null, samples: [], labels: ['惊讶','恐惧','厌恶','高兴','悲伤','愤怒','中性'], videoUrl: null, videoStream: null, videoRunning: false, videoBusy: false, videoTimer: null, videoFrameCount: 0, videoStartedAt: 0, videoNeedsReset: true };
const $ = (id) => document.getElementById(id);
const fmtPct = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`;
const fmtScore = (value) => Number(value || 0).toFixed(3);

function setStatus(kind, text) { $('status-dot').className = kind; $('status-text').textContent = text; }
function showError(message) { $('error-banner').textContent = message; $('error-banner').hidden = false; }
function clearError() { $('error-banner').hidden = true; $('error-banner').textContent = ''; }
function setBusy(busy) { $('run-button').disabled = busy || !(state.selected || state.upload); $('progress-area').hidden = !busy; if (busy) clearError(); }
function setImage(id, src, alt) { $(id).src = src; $(id).alt = alt; }

function chooseSample(sample, button) {
  state.selected = sample; state.upload = null;
  document.querySelectorAll('.sample-button').forEach((el) => el.classList.remove('selected'));
  button?.classList.add('selected');
  $('selected-file').hidden = false; $('file-name').textContent = sample.source; $('file-meta').textContent = `数据集样例 · ${sample.expression} · person ${sample.person}`;
  $('run-button').disabled = false; clearError();
}
function readUpload(file) {
  if (!file) return;
  if (!['image/jpeg','image/png','image/webp'].includes(file.type)) return showError('请选择 JPG、PNG 或 WebP 图片。');
  if (file.size > 10 * 1024 * 1024) return showError('图片需小于 10 MB。');
  const reader = new FileReader(); reader.onload = () => {
    state.upload = { image: reader.result, filename: file.name }; state.selected = null;
    document.querySelectorAll('.sample-button').forEach((el) => el.classList.remove('selected'));
    $('selected-file').hidden = false; $('file-name').textContent = file.name; $('file-meta').textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB · 本机内存中处理`;
    $('run-button').disabled = false; clearError();
  }; reader.readAsDataURL(file);
}

async function loadStatus() {
  try { const data = await (await fetch('/api/status')).json();
    if (data.ready) setStatus('', `${data.hardware} · ${data.device.toUpperCase()}`); else setStatus('error', '模型未就绪');
    if (data.error) showError(data.error);
  } catch { setStatus('error', '本地服务未连接'); showError('无法连接本地模型服务，请确认启动命令仍在运行。'); }
}
async function loadSamples() {
  try { const data = await (await fetch('/api/samples')).json(); state.samples = data.samples || []; $('sample-count').textContent = `${state.samples.length} 类`; const list = $('sample-list'); list.innerHTML = '';
    state.samples.forEach((sample) => { const button = document.createElement('button'); button.type = 'button'; button.className = 'sample-button'; button.innerHTML = `<img src="${sample.image}" alt="${sample.expression}样例"><b>${sample.expression}</b><span>${sample.source.replace('.jpg','').replace('.png','')}</span>`; button.addEventListener('click', () => chooseSample(sample, button)); list.appendChild(button); });
  } catch { $('sample-list').innerHTML = '<div class="loading-line">暂时没有可用样例</div>'; }
}
function renderBars(result) {
  const anon = result.expression_anonymized;
  $('expression-original').textContent = '—'; $('expression-anonymized').textContent = anon.name_zh;
  $('expression-original-confidence').textContent = '原图不进入表情模型'; $('expression-anonymized-confidence').textContent = `匿名图 ${fmtPct(anon.confidence)}`;
  $('expression-status').textContent = '匿名后识别'; $('expression-status').className = 'metric-status pass';
  $('expression-consistency').textContent = '仅匿名图输入'; $('expression-meter').style.width = '100%';
  const hasIdentity = result.identity_prediction_changed !== undefined;
  const originalIdentity = result.identity_original;
  const anonymizedIdentity = result.identity_anonymized;
  $('identity-original').textContent = originalIdentity ? `ID ${originalIdentity.label}` : '未加载';
  $('identity-anonymized').textContent = anonymizedIdentity ? `ID ${anonymizedIdentity.label}` : '未加载';
  $('identity-original-confidence').textContent = originalIdentity ? `置信度 ${fmtPct(originalIdentity.confidence)}` : '—';
  $('identity-anonymized-confidence').textContent = anonymizedIdentity ? `置信度 ${fmtPct(anonymizedIdentity.confidence)}` : '—';
  $('identity-change').textContent = hasIdentity ? (result.identity_prediction_changed ? '是' : '否') : '未加载';
  $('identity-cosine').textContent = hasIdentity ? fmtScore(result.sface_cosine ?? result.identity_cosine) : '—';
  const galleryReady = result.gallery_match_original !== undefined;
  const gallerySize = galleryReady ? Number(result.gallery_size || 0) : 0;
  $('identity-gallery-label').textContent = galleryReady ? `${gallerySize} 人库原身份命中` : 'LFW 身份库原身份命中';
  $('identity-gallery-match').textContent = galleryReady ? (result.gallery_match_original ? '是' : '否') : '未加载';
  $('identity-gallery-confidence').textContent = galleryReady ? `相对置信度 ${fmtPct(result.gallery_confidence)}` : '—';
  $('identity-drop').textContent = hasIdentity ? fmtPct(result.identity_confidence_drop || 0) : '—'; $('identity-meter').style.width = hasIdentity ? `${Math.min(100, (result.identity_confidence_drop || 0) * 100)}%` : '0%';
  $('identity-status').textContent = result.sface_cosine != null ? 'SFace 已计算' : (hasIdentity ? (result.identity_prediction_changed ? '已改变' : '未改变') : '诊断关闭'); $('identity-status').className = `metric-status ${result.sface_cosine != null ? 'pass' : (hasIdentity && result.identity_prediction_changed ? 'pass' : '')}`;
}
function renderResult(result) {
  setImage('original-image', result.original_image, '原始人脸图像'); setImage('anonymized-image', result.anonymized_image, '匿名化人脸图像');
  $('original-caption').textContent = result.filename || '模型输入';
  $('output-caption').textContent = `仅编辑脸部区域 · ${result.resolution.join(' × ')}`;
  $('result-time').textContent = `${result.elapsed_ms} ms`; $('result-device').textContent = result.device.toUpperCase(); $('result-file').textContent = result.filename || '上传图片'; renderBars(result);
  $('results').hidden = false; $('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}
function setVideoSource(file) {
  if (state.videoUrl) URL.revokeObjectURL(state.videoUrl); state.videoUrl = URL.createObjectURL(file); state.videoStream = null;
  const video = $('video-input'); video.srcObject = null; video.src = state.videoUrl; video.load(); $('video-source-name').textContent = file.name; $('video-start').disabled = false; $('video-frame-status').textContent = '视频已就绪';
}
async function openCamera() {
  try { stopVideo(); state.videoStream = await navigator.mediaDevices.getUserMedia({video: {facingMode: 'user'}, audio: false}); const video = $('video-input'); video.src = ''; video.srcObject = state.videoStream; await video.play(); $('video-source-name').textContent = '本机摄像头'; $('video-start').disabled = false; $('video-frame-status').textContent = '摄像头已连接'; }
  catch (error) { showError(`摄像头打开失败：${error.message || '浏览器没有授予摄像头权限'}`); }
}
function stopVideo() {
  state.videoRunning = false; state.videoBusy = false; if (state.videoTimer) { clearTimeout(state.videoTimer); state.videoTimer = null; }
  if (state.videoStream) state.videoStream.getTracks().forEach((track) => track.stop()); state.videoStream = null; $('video-stop').disabled = true; $('video-frame-status').textContent = '已停止';
}
async function processVideoFrame() {
  if (!state.videoRunning) return; const video = $('video-input'), canvas = $('video-output');
  if (video.readyState < 2 || !video.videoWidth) { state.videoTimer = setTimeout(processVideoFrame, 250); return; }
  if (!state.videoBusy) { state.videoBusy = true; const capture = document.createElement('canvas'); const maxWidth = 640; const scale = Math.min(1, maxWidth / video.videoWidth); capture.width = Math.max(1, Math.round(video.videoWidth * scale)); capture.height = Math.max(1, Math.round(video.videoHeight * scale)); capture.getContext('2d').drawImage(video, 0, 0, capture.width, capture.height);
    try { const response = await fetch('/api/process', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({image:capture.toDataURL('image/jpeg', .82), filename:'video-frame.jpg', aligned:false, realtime:true, realtime_reset:state.videoNeedsReset})}); state.videoNeedsReset = false; const result = await response.json(); if (!response.ok) throw new Error(result.error || '视频帧处理失败'); const frameImage = await new Promise((resolve, reject) => { const image = new Image(); image.onload = () => resolve(image); image.onerror = () => reject(new Error('匿名帧解码失败')); image.src = result.anonymized_image; }); canvas.width = frameImage.naturalWidth || frameImage.width; canvas.height = frameImage.naturalHeight || frameImage.height; canvas.getContext('2d').drawImage(frameImage, 0, 0); const exp = result.expression_anonymized; state.videoFrameCount += 1; const elapsed = Math.max(0.001, (performance.now() - state.videoStartedAt) / 1000); $('video-frame-count').textContent = String(state.videoFrameCount); $('video-fps').textContent = `${(state.videoFrameCount / elapsed).toFixed(1)} FPS`; $('video-expression').textContent = exp.name_zh; $('video-confidence').textContent = fmtPct(exp.confidence); $('video-support').textContent = '仅匿名图输入'; $('video-frame-status').textContent = result.background_preserved ? '运行中 · 背景保持' : '运行中'; }
    catch (error) { $('video-frame-status').textContent = error.message || '视频帧处理失败'; } state.videoBusy = false; }
  if (state.videoRunning && !video.ended) state.videoTimer = setTimeout(processVideoFrame, 80); else if (video.ended) stopVideo();
}
function startVideo() { const video = $('video-input'); state.videoRunning = true; state.videoNeedsReset = true; state.videoFrameCount = 0; state.videoStartedAt = performance.now(); $('video-frame-count').textContent = '0'; $('video-fps').textContent = '—'; $('video-stop').disabled = false; video.play().catch(() => {}); processVideoFrame(); }
async function runAnalysis() {
  const payload = state.selected ? { sample_id: state.selected.id } : { ...state.upload, aligned: $('aligned-input').checked };
  setBusy(true); $('progress-text').textContent = '正在执行人脸匿名化与多模型验收…';
  try { const response = await fetch('/api/process', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload) }); const result = await response.json(); if (!response.ok) throw new Error(result.error || '处理失败'); renderResult(result); }
  catch (error) { showError(error.message || '模型处理失败'); } finally { setBusy(false); }
}
function renderReport(data) {
  if (!data.available) { $('report-grid').innerHTML = '<div class="loading-line">暂无 evaluation.json，请先运行联合评估。</div>'; $('report-updated').textContent = 'NO REPORT'; return; }
  const s = data.summary; $('report-updated').textContent = `更新于 ${data.updated_at.replace('T',' ')}`;
  const identityChange = s.identity_prediction_change_rate == null ? '未启用' : fmtPct(s.identity_prediction_change_rate);
  const identityAccuracy = s.identity_anonymized_accuracy == null ? '未启用' : fmtPct(s.identity_anonymized_accuracy);
  const sfaceSimilarity = s.sface_mean_cosine == null ? '未启用' : fmtScore(s.sface_mean_cosine);
  const galleryMatch = s.gallery_original_match_rate == null ? '未启用' : fmtPct(s.gallery_original_match_rate);
  const identityNote = s.identity_prediction_change_rate == null ? '当前报告未加载身份诊断' : '离线身份诊断';
  const gallerySize = Number(s.gallery_size || 0);
  const galleryTitle = gallerySize ? `${gallerySize} 人库原身份命中率` : 'LFW 人库原身份命中率';
  const galleryNote = gallerySize ? `原脸 + ${Math.max(0, gallerySize - 1)} 个干扰身份` : '原脸 + LFW 干扰身份';
  const cards = [['匿名图表情准确率', fmtPct(s.expression_anonymized_accuracy), '表情模型只接收匿名图', false], ['表情模型输入', '匿名图', '原始图不进入表情模型', false], ['局部编辑样本数', String(s.samples), '只改检测到的人脸区域', false], ['SFace 平均脸相似度', sfaceSimilarity, '身份诊断主指标', true], [galleryTitle, galleryMatch, galleryNote, true], ['闭集身份准确率', identityAccuracy, '仅作回归对照', true], ['闭集预测变化率', identityChange, '只看紧裁脸部', true]];
  $('report-grid').innerHTML = cards.map(([title, value, note, accent]) => `<article class="report-card ${accent ? 'accent' : ''}"><small>${title}</small><strong>${value}</strong><small>${note}</small></article>`).join(''); $('report-scope').textContent = `${data.scope} · 样本数 ${s.samples}`;
}
async function loadReport() { try { renderReport(await (await fetch('/api/evaluation')).json()); } catch { $('report-grid').innerHTML = '<div class="loading-line">评估报告读取失败</div>'; } }

$('dropzone').addEventListener('click', () => $('file-input').click()); $('dropzone').addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') $('file-input').click(); });
$('dropzone').addEventListener('dragover', (e) => { e.preventDefault(); $('dropzone').classList.add('selected'); }); $('dropzone').addEventListener('dragleave', () => $('dropzone').classList.remove('selected')); $('dropzone').addEventListener('drop', (e) => { e.preventDefault(); $('dropzone').classList.remove('selected'); readUpload(e.dataTransfer.files[0]); });
$('file-input').addEventListener('change', (e) => readUpload(e.target.files[0])); $('run-button').addEventListener('click', runAnalysis); $('clear-file').addEventListener('click', () => { state.selected = null; state.upload = null; $('selected-file').hidden = true; $('file-input').value = ''; $('run-button').disabled = true; document.querySelectorAll('.sample-button').forEach((el) => el.classList.remove('selected')); });
$('video-file-input').addEventListener('change', (e) => { if (e.target.files[0]) setVideoSource(e.target.files[0]); }); $('camera-button').addEventListener('click', openCamera); $('video-start').addEventListener('click', startVideo); $('video-stop').addEventListener('click', stopVideo);
loadStatus(); loadSamples(); loadReport();
