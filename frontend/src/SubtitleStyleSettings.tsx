import { useState, type CSSProperties, type ReactNode } from "react";

export type SubtitleStyleOptions = {
  font_name: string;
  font_size: number;
  subtitle_color: string;
  subtitle_outline_color: string;
  subtitle_effect: string;
  subtitle_placement: "overlay" | "bottom-bar";
  subtitle_margin_ratio: number;
  subtitle_display_mode: string;
  subtitle_preset: string;
};

const DEFAULT_FONT = "Microsoft YaHei";
const FONTS = [
  [DEFAULT_FONT, "微软雅黑"],
  ["DengXian", "等线"],
  ["SimHei", "黑体"],
  ["SimSun", "宋体"],
  ["KaiTi", "楷体"],
  ["FangSong", "仿宋"],
  ["Arial", "Arial"],
];
const CUSTOM_FONT = "__custom__";

type SubtitleAppearance = Pick<SubtitleStyleOptions,
  "font_name" | "font_size" | "subtitle_color" | "subtitle_outline_color" | "subtitle_effect"
>;

const STYLES: { label: string; appearance: SubtitleAppearance }[] = [
  { label: "清晰白字", appearance: { font_name: DEFAULT_FONT, font_size: 24, subtitle_color: "白色", subtitle_outline_color: "黑色", subtitle_effect: "描边" } },
  { label: "综艺黄字", appearance: { font_name: "SimHei", font_size: 28, subtitle_color: "黄色", subtitle_outline_color: "黑色", subtitle_effect: "描边+阴影" } },
  { label: "楷体手写", appearance: { font_name: "KaiTi", font_size: 27, subtitle_color: "白色", subtitle_outline_color: "黑色", subtitle_effect: "阴影" } },
  { label: "宋体纪录", appearance: { font_name: "SimSun", font_size: 25, subtitle_color: "白色", subtitle_outline_color: "黑色", subtitle_effect: "描边+阴影" } },
];

// Use the same palette and bilingual proportions as the ASS renderer.
const COLORS: Record<string, string> = {
  白色: "#ffffff", 黄色: "#ffff00", 青色: "#00ffff", 绿色: "#00ff00",
  黑色: "#000000", 灰色: "#808080", 蓝色: "#0000ff",
};
const TEXT_COLORS = ["白色", "黄色", "青色", "绿色", "黑色"];
const OUTLINE_COLORS = ["黑色", "白色", "灰色", "蓝色"];
const EFFECTS = ["描边", "阴影", "描边+阴影", "无"];
const MARGINS = [
  { value: 0.03, label: "3%（贴近底部）" },
  { value: 0.055, label: "5.5%（推荐）" },
  { value: 0.1, label: "10%（偏上）" },
  { value: 0.15, label: "15%（更高）" },
];
const SOURCE_STYLES: Record<string, CSSProperties> = {
  soft: { color: "#c8c8c8", fontSize: "0.85em" },
  compact: { color: "#a0a0a0", fontSize: "0.7em" },
  equal: {},
};

function subtitlePaint(appearance: SubtitleAppearance): CSSProperties {
  return {
    fontFamily: `${JSON.stringify(appearance.font_name.trim() || DEFAULT_FONT)}, "${DEFAULT_FONT}", sans-serif`,
    color: COLORS[appearance.subtitle_color],
    WebkitTextStroke: appearance.subtitle_effect.includes("描边") ? `1.5px ${COLORS[appearance.subtitle_outline_color]}` : undefined,
    textShadow: appearance.subtitle_effect.includes("阴影") ? "1.5px 1.5px 1px #000" : undefined,
  };
}

export function SubtitleStyleSettings({ options, onChange, fontError, effectGuide, untranslated = false }: {
  options: SubtitleStyleOptions;
  onChange: (patch: Partial<SubtitleStyleOptions>) => void;
  fontError?: string;
  effectGuide?: ReactNode;
  untranslated?: boolean;
}) {
  const [typedFont, setTypedFont] = useState<string | null>(null);
  const currentFont = options.font_name.trim();
  const usesCustomFont = typedFont === options.font_name || !FONTS.some(([name]) => name === currentFont);
  const selectedFont = currentFont && usesCustomFont ? CUSTOM_FONT : currentFont || DEFAULT_FONT;
  const previewFontSize = Math.round(10 + Math.min(96, Math.max(8, options.font_size)) * 0.41);
  const sourceLine = <span style={SOURCE_STYLES[options.subtitle_preset]}>Preview</span>;

  return (
    <div className="subtitle-style-settings">
      <div className="subtitle-style-presets" role="group" aria-label="一键样式">
        <strong className="subtitle-style-heading">一键样式</strong>
        <div className="subtitle-style-choices">
          {STYLES.map(({ label, appearance }) => (
            <button
              type="button"
              key={label}
              aria-pressed={Object.entries(appearance).every(([key, value]) => options[key as keyof SubtitleAppearance] === value)}
              onClick={() => {
                setTypedFont(null);
                onChange(appearance);
              }}
            >
              <span className="subtitle-style-swatch" style={subtitlePaint(appearance)} aria-hidden="true">字幕</span>
              {label}
            </button>
          ))}
        </div>
      </div>
      <figure className="subtitle-style-preview">
        <figcaption><strong className="subtitle-style-heading">实时预览</strong><small>放大示例</small></figcaption>
        <div className={`subtitle-style-frame${options.subtitle_placement === "bottom-bar" ? " with-subtitle-bar" : ""}`}>
          <div className="subtitle-style-lines" style={{
            ...subtitlePaint(options),
            fontSize: `min(${previewFontSize}px, 18cqw)`,
            bottom: `${options.subtitle_margin_ratio * 100}%`,
          }}>
            {options.subtitle_display_mode === "bilingual-source-first" && sourceLine}
            <span>字幕示例</span>
            {options.subtitle_display_mode === "bilingual-translation-first" && sourceLine}
          </div>
        </div>
      </figure>
      <div className="settings-grid subtitle-style-fields">
        <div className="subtitle-font-settings">
          <label>
            常用字体
            <select value={selectedFont} onChange={(event) => {
              setTypedFont(null);
              onChange({ font_name: event.target.value });
            }}>
              {FONTS.map(([name, label]) => <option key={name} value={name}>{label}</option>)}
              {selectedFont === CUSTOM_FONT && <option value={CUSTOM_FONT} disabled>已使用自定义字体</option>}
            </select>
          </label>
          <label>
            自定义字体名称
            <input
              value={usesCustomFont ? options.font_name : ""}
              maxLength={100}
              placeholder="输入已安装的字体名称"
              onChange={(event) => {
                setTypedFont(event.target.value);
                onChange({ font_name: event.target.value });
              }}
            />
            {fontError && <small className="field-error">{fontError}</small>}
          </label>
          <div className="subtitle-font-sources">
            <strong>查找字体</strong>
            <div className="subtitle-font-actions">
              <a className="subtitle-font-download" href="https://www.maoken.com/freefonts" target="_blank" rel="noreferrer">
                浏览中文字体库 <span aria-hidden="true">↗</span>
              </a>
              <a href="ms-settings:fonts">查看已装字体</a>
              <a href="ms-windows-store://collection/?collectionid=fonts">微软商店</a>
            </div>
            <small>下载后右键安装，再填写字体名称。</small>
          </div>
        </div>
        <label>
          字号
          <input type="number" min="8" max="96" step="1" value={options.font_size} onChange={(event) => onChange({ font_size: Number(event.target.value || 24) })} />
        </label>
        <label>
          字幕颜色
          <select value={options.subtitle_color} onChange={(event) => onChange({ subtitle_color: event.target.value })}>
            {TEXT_COLORS.map((color) => <option key={color}>{color}</option>)}
          </select>
        </label>
        <label>
          描边颜色
          <select value={options.subtitle_outline_color} onChange={(event) => onChange({ subtitle_outline_color: event.target.value })}>
            {OUTLINE_COLORS.map((color) => <option key={color}>{color}</option>)}
          </select>
        </label>
        <label>
          字幕效果
          <select value={options.subtitle_effect} onChange={(event) => onChange({ subtitle_effect: event.target.value })}>
            {EFFECTS.map((effect) => <option key={effect}>{effect}</option>)}
          </select>
          {effectGuide}
        </label>
        <label>
          字幕摆放
          <select value={options.subtitle_placement} onChange={(event) => onChange({ subtitle_placement: event.target.value as SubtitleStyleOptions["subtitle_placement"] })}>
            <option value="overlay">叠在画面上</option>
            <option value="bottom-bar">画面外黑色字幕栏</option>
          </select>
        </label>
        <label>
          字幕底部位置
          <select value={options.subtitle_margin_ratio} onChange={(event) => onChange({ subtitle_margin_ratio: Number(event.target.value) })}>
            {MARGINS.map(({ value, label }) => <option key={value} value={value}>{label}</option>)}
            {!MARGINS.some(({ value }) => value === options.subtitle_margin_ratio) && <option value={options.subtitle_margin_ratio}>
              {Number((options.subtitle_margin_ratio * 100).toFixed(1))}%（当前）
            </option>}
          </select>
        </label>
        <label>
          字幕排版
          <select value={options.subtitle_display_mode} onChange={(event) => onChange({ subtitle_display_mode: event.target.value })}>
            <option value="translated">{untranslated ? "原文字幕" : "译文单语"}</option>
            <option value="bilingual-source-first">原文在上 + 译文在下</option>
            <option value="bilingual-translation-first">译文在上 + 原文在下</option>
          </select>
        </label>
        <label>
          双语样式
          <select value={options.subtitle_preset} disabled={options.subtitle_display_mode === "translated"} onChange={(event) => onChange({ subtitle_preset: event.target.value })}>
            <option value="soft">弱化原文（小一号、浅灰）</option>
            <option value="compact">小字原文（更小、更灰）</option>
            <option value="equal">两行一致</option>
          </select>
        </label>
      </div>
    </div>
  );
}
