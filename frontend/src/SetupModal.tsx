export type TranslationPreset = {
  id: string;
  label: string;
  group: "online" | "local";
  engine: "deepseek" | "openai" | "ollama";
  needs_key: boolean;
  key_optional: boolean;
  base_url: string;
  base_editable: boolean;
  model: string;
  model_editable: boolean;
  model_choices: { id: string; label: string }[];
  key_url: string;
  key_placeholder: string;
  requirements: string[];
};

type FieldsProps = {
  presets: TranslationPreset[];
  presetId: string;
  baseUrl: string;
  model: string;
  apiKey: string;
  disabled?: boolean;
  onPresetChange: (id: string) => void;
  onBaseChange: (value: string) => void;
  onModelChange: (value: string) => void;
  onKeyChange: (value: string) => void;
  onOpenKeyPage: () => void;
};

export function TranslationPresetFields({
  presets, presetId, baseUrl, model, apiKey, disabled = false,
  onPresetChange, onBaseChange, onModelChange, onKeyChange, onOpenKeyPage,
}: FieldsProps) {
  const preset = presets.find((item) => item.id === presetId) ?? presets[0];
  if (!preset) return null;
  const online = presets.filter((item) => item.group === "online");
  const local = presets.filter((item) => item.group === "local");
  return (
    <>
      <label>
        翻译服务
        <select value={preset.id} disabled={disabled} onChange={(event) => onPresetChange(event.target.value)}>
          <optgroup label="在线服务">
            {online.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
          </optgroup>
          <optgroup label="本机部署">
            {local.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
          </optgroup>
        </select>
      </label>
      <div className="setup-requirements" aria-label="配置要求">
        {preset.requirements.map((line) => <p key={line}>{line}</p>)}
      </div>
      {preset.base_editable ? (
        <label>
          接口地址
          <input value={baseUrl} disabled={disabled} placeholder="例如 http://127.0.0.1:1234/v1" onChange={(event) => onBaseChange(event.target.value)} />
        </label>
      ) : preset.base_url ? <p className="drawer-note">接口地址：{preset.base_url}</p> : null}
      <label>
        模型名
        {preset.model_choices.length > 0 ? (
          <select
            value={preset.model_choices.some((item) => item.id === model) ? model : "__custom__"}
            disabled={disabled}
            onChange={(event) => onModelChange(event.target.value === "__custom__" ? "" : event.target.value)}
          >
            {preset.model_choices.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
            {preset.model_editable && <option value="__custom__">本机已经有的其他模型</option>}
          </select>
        ) : (
          <input
            value={model}
            maxLength={128}
            disabled={disabled || !preset.model_editable}
            placeholder={preset.model || "按该服务文档填写模型 ID"}
            onChange={(event) => onModelChange(event.target.value)}
          />
        )}
      </label>
      {preset.model_choices.length > 0 && preset.model_editable && !preset.model_choices.some((item) => item.id === model) && (
        <label>
          已有模型名
          <input
            value={model}
            maxLength={128}
            disabled={disabled}
            placeholder="例如 llama3.1:8b，只填这一个"
            onChange={(event) => onModelChange(event.target.value)}
          />
        </label>
      )}
      {preset.needs_key && (
        <label className="setup-key">
          {preset.key_optional ? "API Key（可留空）" : "API Key"}
          <input
            type="password"
            autoComplete="off"
            value={apiKey}
            maxLength={1024}
            placeholder={preset.key_placeholder}
            disabled={disabled}
            onChange={(event) => onKeyChange(event.target.value)}
          />
          {preset.key_url && (
            <span className="setup-key-actions">
              <button type="button" className="settings-inline-link" disabled={disabled} onClick={onOpenKeyPage}>去申请密钥</button>
            </span>
          )}
        </label>
      )}
    </>
  );
}

type SetupModalProps = {
  presets: TranslationPreset[];
  presetId: string;
  baseUrl: string;
  model: string;
  ffmpegReady: boolean;
  modelReady: boolean;
  keyValue: string;
  busy: boolean;
  active: "ffmpeg" | "whisper" | "";
  progress: string;
  error: string;
  onPresetChange: (id: string) => void;
  onBaseChange: (value: string) => void;
  onModelChange: (value: string) => void;
  onKeyChange: (value: string) => void;
  onOpenKeyPage: () => void;
  onInstall: () => void;
  onSkip: () => void;
};

export function SetupModal({
  presets, presetId, baseUrl, model, ffmpegReady, modelReady, keyValue, busy, active, progress, error,
  onPresetChange, onBaseChange, onModelChange, onKeyChange, onOpenKeyPage, onInstall, onSkip,
}: SetupModalProps) {
  const preset = presets.find((item) => item.id === presetId);
  const keySatisfied = !preset || !preset.needs_key || preset.key_optional || keyValue.trim().length > 0;
  const ready = ffmpegReady && modelReady && keySatisfied;
  return (
    <div className="modal-backdrop" role="presentation">
      <section className="template-modal setup-modal" role="dialog" aria-modal="true" aria-label="第一次准备">
        <header>
          <div>
            <p className="eyebrow">第一次打开</p>
            <h2>先准备这台电脑</h2>
          </div>
        </header>
        <p className="setup-lead">补上播放处理组件、一个语音模型，并选好翻译服务。在线服务要按说明填写密钥；本机服务要先把程序跑起来。装好或跳过之后，这个页面不会再出现。</p>
        <ul className="setup-rows">
          <li><b>FFmpeg</b><span>{ffmpegReady ? "已就绪" : active === "ffmpeg" ? progress || "正在下载" : "将自动下载"}</span></li>
          <li><b>语音模型</b><span>{modelReady ? "已就绪" : active === "whisper" ? progress || "正在下载" : "将自动下载，约 500 MB"}</span></li>
          <li><b>{preset?.label ?? "翻译服务"}</b><span>{keySatisfied ? "可以保存" : "还要按下面的要求填写"}</span></li>
        </ul>
        <TranslationPresetFields
          presets={presets}
          presetId={presetId}
          baseUrl={baseUrl}
          model={model}
          apiKey={keyValue}
          disabled={busy}
          onPresetChange={onPresetChange}
          onBaseChange={onBaseChange}
          onModelChange={onModelChange}
          onKeyChange={onKeyChange}
          onOpenKeyPage={onOpenKeyPage}
        />
        {error && <p className="setup-error">{error}</p>}
        <footer>
          <button type="button" disabled={busy} onClick={onSkip}>以后再看</button>
          <button type="button" className="primary" disabled={busy} onClick={onInstall}>{busy ? "正在准备…" : ready ? "保存并开始" : "补齐并开始"}</button>
        </footer>
      </section>
    </div>
  );
}
