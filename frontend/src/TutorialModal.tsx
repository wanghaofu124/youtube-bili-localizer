import { useEffect, useRef, useState, type ReactNode } from "react";

export type TutorialAction = "material" | "process" | "dependencies" | "settings" | "subtitle" | "publish" | "files";

type TutorialChapter = {
  id: string;
  title: string;
  eyebrow: string;
  body: ReactNode;
  figure?: { src: string; alt: string; caption: string };
  action?: { label: string; target: TutorialAction };
};

const Steps = ({ items }: { items: string[] }) => <ol className="tutorial-steps">{items.map((item) => <li key={item}>{item}</li>)}</ol>;
const Tip = ({ children }: { children: ReactNode }) => <p className="tutorial-tip">{children}</p>;
const ChoiceTable = ({ rows }: { rows: Array<[string, string, string]> }) => <div className="tutorial-choice-table"><div className="tutorial-choice-head"><b>选择</b><b>适合</b><b>注意</b></div>{rows.map(([choice, fit, note]) => <div className="tutorial-choice-row" key={choice}><b>{choice}</b><span>{fit}</span><span>{note}</span></div>)}</div>;

const chapters: TutorialChapter[] = [
  { id: "overview", title: "认识工作流", eyebrow: "开始 · 先看路线", body: <><p>工作台按五个阶段保存检查点：获取素材、字幕提取、翻译、渲染、发布辅助。每一阶段完成后才进入下一阶段，失败或关闭应用后可从最后有效检查点继续。</p><Steps items={["准备一段你有权处理的素材；第一次建议使用“授权演示结果”或约 10 秒片段。", "在“素材”页选择 URL 或本地文件，确认授权和处理范围。", "保存方案并检查准备，解决阻断项后运行需要的阶段。", "在“字幕”页人工检查，再重新渲染并到“文件”页取出成片。", "需要投稿时进入“发布”页准备草稿，最终公开提交由你手动确认。"]} /><Tip>打开教程、点击章节导航或跳转页面都不会自动开始处理、安装、下载或发送在线请求。</Tip></>, figure: { src: "/images/tutorial/tutorial-material.jpg", alt: "素材页显示视频来源、授权和处理范围设置", caption: "在“处理范围”选择“仅处理前10秒”试跑；确认授权后保存素材并配置方案。" } },
  { id: "material", title: "添加素材与范围", eyebrow: "素材 · 先确认授权", body: <><p>“素材”页负责登记来源、范围和画质。URL 会先读取标题与时长；本地文件直接登记原路径，不会因为打开教程而复制或上传。</p><Steps items={["点击左侧“素材”，选择“视频链接”或“本地文件”。", "粘贴 YouTube/B 站链接，或选择本地视频；确认授权复选框。", "读取元数据后，在“处理范围”选择完整视频或仅处理前，并输入秒/分钟/小时范围。", "在画质和访问设置中选择需要的质量；高画质会增加网络、磁盘和处理时间。", "点击“保存素材并配置方案”，只保存方案并进入准备检查。"]} /><Tip>常见错误：链接读不到时先检查访问设置、Cookies 或平台可见性；本地原文件移动后，应用可能无法继续读取；不要把别人的 Cookies 粘贴到公开 Issue。</Tip></>, figure: { src: "/images/tutorial/tutorial-material.jpg", alt: "素材页显示视频来源、授权和处理范围设置", caption: "在“处理范围”选择“仅处理前10秒”试跑；确认授权后保存素材并配置方案。" }, action: { label: "打开素材页", target: "material" } },
  { id: "prepare", title: "准备工具与模型", eyebrow: "依赖中心 · 按需准备", body: <><p>依赖中心按当前方案显示需要的组件。WebView2 用于桌面界面，FFmpeg 用于音频/视频处理，Whisper 用于语音转写，Tesseract 用于 OCR，Node.js/浏览器用于部分平台验证和投稿辅助，Ollama 运行本机翻译模型。</p><Steps items={["从顶部“更多”打开“依赖中心”，先点“重新检测”。", "按字幕来源安装必需组件：音频转写需要 Whisper/FFmpeg，OCR 需要 Tesseract 语言包。", "使用在线翻译时配置对应服务；使用本机翻译时启动 Ollama 并选择模型。", "查看组件状态变为可用，再回到“处理”页运行准备检查。"]} /><ChoiceTable rows={[["small Whisper", "普通桌面、先跑通流程", "更快更省资源，仍需人工复核"],["large-v3 Whisper", "有更多内存/显存、重视转写质量", "更慢、更占资源，仍不是免校对"],["Ollama", "希望文本请求留在本机", "无在线翻译 API 费，但占内存/显存/磁盘"],["DeepSeek/OpenAI", "需要在线翻译或文案生成", "请求前会提示内容和费用，账单以服务商为准"]]} /><Tip>不要为了“自动完成”安装所有组件；先按字幕来源和目标功能选择，缺少可选组件不必处理。</Tip></>, figure: { src: "/images/tutorial/tutorial-dependencies.jpg", alt: "依赖中心显示 FFmpeg 和 Whisper 检测与安装入口", caption: "“可用”表示组件已就绪；按需要安装 Whisper 模型，不必全部安装，状态以本机检测为准。" }, action: { label: "打开依赖中心", target: "dependencies" } },
  { id: "translation", title: "选择转写与翻译", eyebrow: "处理方案 · 区分来源与服务", body: <><p>GPU/CUDA 主要影响 Whisper 语音转写速度；它不等于 OCR，也不决定翻译服务。字幕来源决定是否读取音频、画面文字或平台字幕。</p><Steps items={["打开“处理”，找到“字幕来源”：自动、音频、OCR 或音频 + OCR 合并。", "选择 Whisper 模型、源语言和设备；CPU 更通用，CUDA 需要可用的 NVIDIA 驱动。", "在“翻译服务与模型”选择 DeepSeek、OpenAI、Ollama 或“不翻译”。", "设置目标语言和智能上下文；在线服务发送前会显示服务商、端点、模型、内容类别和费用提示。", "点击“保存方案并检查准备”，确认检查结果后再运行。"]} /><Tip>先排查字幕提取可以选“不翻译”；Ollama 无在线翻译 API 费但需要本机模型，在线服务费用估算只供参考，最终以账单为准。</Tip></>, figure: { src: "/images/tutorial/tutorial-process.jpg", alt: "处理页显示字幕来源、Whisper 和翻译器选择", caption: "左侧选择字幕来源/Whisper，右侧选择翻译器；图中 DeepSeek 缺少 Key 时请配置或更换翻译器。" }, action: { label: "打开处理页", target: "process" } },
  { id: "run", title: "运行与失败恢复", eyebrow: "检查点 · 出错后继续", body: <><p>运行前准备检查会验证依赖、输入和配置。运行中不要反复点击主操作；任务详情会显示当前阶段、百分比、费用小表和日志。</p><Steps items={["点击“保存方案并检查准备”，逐项查看通过、警告和阻断。", "解决阻断项后，从“运行当前阶段”或顶部主操作开始；按需要运行到渲染。", "失败时打开右侧“任务详情”，查看阶段错误和“完整日志”；依赖问题可回到依赖中心重新检测。", "修复设置后回到同一任务，点击“从检查点继续”或运行失败阶段。", "确认已完成的阶段保持有效，再继续后续阶段；不会无故重新下载或转写全部内容。"]} /><Tip>任务库中的恢复任务不会自动消耗资源；载入后仍需你确认准备检查和下一阶段。</Tip></>, figure: { src: "/images/tutorial/tutorial-process.jpg", alt: "处理页显示保存方案和准备检查入口", caption: "先保存方案并检查准备；失败后查看完整日志，再从失败检查点继续。" }, action: { label: "打开处理页", target: "process" } },
  { id: "subtitle", title: "人工校对字幕", eyebrow: "字幕校对 · 先试听再编辑", body: <><p>字幕页的目标是让你确认整份字幕同步、文字和问题标记。自动候选只提供建议，不代替人工决定。</p><Steps items={["打开“字幕”，先用“问题筛选”切换“全部字幕/待复核”。", "点击字幕行左侧的序号按钮（播放第 N 段字幕）试听这一条并检查时间轴。", "直接编辑原文或译文；单条重译后点“采用候选”，批量精简使用“生成精简建议”再点“采用选中”。", "不满意时用撤销恢复编辑；批量替换先看预览，再应用到选定范围。", "点击“保存字幕”，确认保存成功；字幕页按 Ctrl+S 保存，输入框内也可保存，Alt↑↓ 与空格仅在未编辑文字时使用。", "字幕文字、时间轴或样式改动后，到处理/字幕流程重新执行渲染。"]} /><Tip>保存字幕 ≠ 重新渲染视频。整份字幕同步调整应先保存并检查，再重新生成成片。</Tip></>, figure: { src: "/images/tutorial/tutorial-subtitles.jpg", alt: "字幕页显示字幕行、试听按钮和校对区域", caption: "只读演示：用来定位序号、问题筛选、保存和重渲染；载入实际已完成任务后可编辑。" }, action: { label: "打开字幕页", target: "subtitle" } },
  { id: "render", title: "渲染与导出", eyebrow: "成片文件 · 渲染后检查", body: <><p>字幕页的样式和时间轴设置决定成片外观。流程会保存 SRT/字幕文件；FFmpeg 负责生成硬字幕成片。渲染不是简单“导出草稿”，而是重新执行视频处理阶段。</p><Steps items={["在“字幕”页查看原片/成片预览，确认字幕内容和整份时间轴。", "在字幕样式设置中调整字号、字幕排版、颜色、描边/阴影和底部位置。", "选择自动、NVIDIA NVENC 或 CPU 编码；自动优先可用的 NVIDIA，失败时可用 CPU。", "运行“渲染”阶段，等待成片检查点完成；修改样式或字幕后要再次渲染。", "打开“文件”页，查看 SRT、字幕和硬字幕视频，确认输出目录与文件大小。"]} /><Tip>“文件”页负责查看、打开和删除已生成输出；删除前会显示预览。渲染完成不等于已经投稿。</Tip></>, figure: { src: "/images/tutorial/tutorial-render.jpg", alt: "字幕样式和输出编码设置区域", caption: "在“字幕样式”调整字号/排版/颜色，在“字幕渲染”选择编码器，之后重新渲染更新成片。" }, action: { label: "打开文件页", target: "files" } },
  { id: "publish", title: "B 站与抖音投稿辅助", eyebrow: "发布准备 · 草稿仍需确认", body: <><p>发布页用于准备标题、简介、标签、封面和成片，并辅助打开 B 站或抖音页面。它会停在最终公开提交前，但辅助操作可能上传你选择的文件并填写草稿；最终公开投稿始终由你手动确认。</p><Steps items={["打开“发布”，选择要处理的成片和封面，检查目标平台。", "检查标题、简介、标签和模板；需要生成式文案时，先阅读在线服务提示。", "B 站或抖音辅助流程会打开相应浏览器，并可能上传你选择的视频/封面、填写草稿字段。", "在浏览器中核对账号、可见范围、分区和内容，再决定是否提交。", "若只想准备素材而不上传，停在发布页或使用复制按钮，不启动辅助上传。"]} /><Tip>Cookie、浏览器 Profile、API Key 和字幕原文都是敏感信息；不要放入公开截图或 Issue。</Tip></>, figure: { src: "/images/tutorial/tutorial-publish.jpg", alt: "发布页展示成片、封面和投稿信息填写顺序", caption: "按平台→成片→文案→辅助填表操作；本图尚未选成片，按钮暂不可用，最终公开投稿由你手动确认。" }, action: { label: "打开发布页", target: "publish" } },
  { id: "faq", title: "常见问题与隐私", eyebrow: "查阅 · 处理边界", body: <><p><b>没有 API Key：</b>处理页可选 Ollama 或“不翻译”，也可先查看授权演示。<br /><b>模型很慢：</b>检查模型大小、CPU/CUDA、资源模式和磁盘空间。<br /><b>字幕改完没变化：</b>保存字幕后仍需重新渲染。<br /><b>任务失败：</b>先看任务详情完整日志和依赖中心，再从失败检查点继续。<br /><b>投稿担心误发：</b>发布辅助可能上传并填写草稿，最终公开提交仍由你手动确认。</p><Steps items={["遇到问题先记录当前页面、阶段和错误摘要，不要复制密钥、Cookies 或完整字幕。", "检查“任务详情”的完整日志，再打开依赖中心重新检测。", "确认本机数据目录、输出目录和服务商账单分别由应用、Windows 文件系统和服务商管理。"]} /><Tip>视频、字幕、日志、缓存、模型和输出默认保存在本机；在线平台和 AI 服务按你选择并确认的操作处理请求。公开反馈请使用 GitHub Issues，勿粘贴敏感信息。</Tip></> },
];

type TutorialModalProps = {
  onClose: () => void;
  onAction: (action: TutorialAction) => void;
};

export function TutorialModal({ onClose, onAction }: TutorialModalProps) {
  const [activeId, setActiveId] = useState(chapters[0].id);
  const [imageViewer, setImageViewer] = useState<TutorialChapter["figure"] | null>(null);
  const [imageFit, setImageFit] = useState<"fit" | "original">("fit");
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const imageCloseButtonRef = useRef<HTMLButtonElement>(null);
  const imageTriggerRef = useRef<HTMLButtonElement>(null);
  const dialogRef = useRef<HTMLElement>(null);
  const active = chapters.find((chapter) => chapter.id === activeId) ?? chapters[0];

  useEffect(() => {
    closeButtonRef.current?.focus();
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        if (imageViewer) {
          setImageViewer(null);
          window.setTimeout(() => imageTriggerRef.current?.focus(), 0);
        } else onClose();
        return;
      }
      if (event.key !== "Tab" || !dialogRef.current) return;
      const focusable = Array.from(dialogRef.current.querySelectorAll<HTMLElement>("button, [href], input, select, textarea, [tabindex]:not([tabindex='-1'])")).filter((item) => !item.hasAttribute("disabled"));
      if (!focusable.length) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    };
    document.addEventListener("keydown", onKeyDown, true);
    return () => document.removeEventListener("keydown", onKeyDown, true);
  }, [imageViewer, onClose]);

  useEffect(() => {
    if (imageViewer) imageCloseButtonRef.current?.focus();
  }, [imageViewer]);

  const closeImageViewer = () => {
    setImageViewer(null);
    window.setTimeout(() => imageTriggerRef.current?.focus(), 0);
  };
  const openImageViewer = (figure: TutorialChapter["figure"]) => {
    setImageFit("fit");
    setImageViewer(figure);
  };

  const runAction = (action: TutorialAction) => {
    onClose();
    window.setTimeout(() => onAction(action), 0);
  };

  return <div className="modal-backdrop tutorial-backdrop" role="presentation" onMouseDown={(event) => event.stopPropagation()}>
    <section ref={dialogRef} className={`template-modal tutorial-modal${imageViewer ? " tutorial-modal-image" : ""}`} role="dialog" aria-modal="true" aria-labelledby="tutorial-title">
      <header className="tutorial-header">
        <div><p className="eyebrow">新手教程</p><h2 id="tutorial-title">从视频到中文字幕成片</h2><p>按真实工作台流程了解每一步；教程不会启动任务、下载或调用在线服务。</p></div>
        <button ref={closeButtonRef} onClick={onClose} aria-label="关闭新手教程">×</button>
      </header>
      <div className={`tutorial-layout${imageViewer ? " tutorial-image-layout" : ""}`}>
        {!imageViewer && <nav className="tutorial-chapters" aria-label="教程章节">
          <label className="tutorial-select-label">章节<select value={active.id} onChange={(event) => setActiveId(event.target.value)}>{chapters.map((chapter) => <option key={chapter.id} value={chapter.id}>{chapter.title}</option>)}</select></label>
          <div className="tutorial-chapter-list">{chapters.map((chapter, index) => <button key={chapter.id} className={chapter.id === active.id ? "active" : ""} onClick={() => setActiveId(chapter.id)}><span>{String(index + 1).padStart(2, "0")}</span><b>{chapter.title}</b></button>)}</div>
        </nav>}
        <article className="tutorial-content">
          {imageViewer ? <div className="tutorial-image-viewer">
            <div className="tutorial-image-viewer-head"><div><p className="eyebrow">图片查看</p><h3>{active.title}</h3></div><button ref={imageCloseButtonRef} onClick={closeImageViewer} aria-label="返回教程章节">×</button></div>
            <div className="tutorial-image-tools"><button className={imageFit === "fit" ? "active" : ""} onClick={() => setImageFit("fit")}>适应窗口</button><button className={imageFit === "original" ? "active" : ""} onClick={() => setImageFit("original")}>原始尺寸</button></div>
            <div className={`tutorial-image-scroll${imageFit === "original" ? " original" : ""}`}><img src={imageViewer.src} alt={imageViewer.alt} /></div>
            <p className="tutorial-image-caption">{imageViewer.caption} 原始尺寸可滚动查看细节。</p>
            <button className="secondary tutorial-image-back" onClick={closeImageViewer}>返回本章</button>
          </div> : <>
            <p className="eyebrow">{active.eyebrow}</p>
            <h3>{active.title}</h3>
            <div className="tutorial-copy">{active.body}</div>
            {active.figure && <figure className="tutorial-figure"><button ref={imageTriggerRef} className="tutorial-image-button" onClick={() => openImageViewer(active.figure!)} aria-label={`放大查看：${active.figure.alt}`}><img src={active.figure.src} alt={active.figure.alt} loading="lazy" /></button><figcaption>{active.figure.caption}<span>点击图片放大</span></figcaption></figure>}
            {active.action && <button className="primary tutorial-action" onClick={() => runAction(active.action!.target)}>{active.action.label} →</button>}
            <div className="tutorial-pager"><button onClick={() => setActiveId(chapters[Math.max(0, chapters.findIndex((item) => item.id === active.id) - 1)].id)} disabled={active.id === chapters[0].id}>上一章</button><span>第 {chapters.findIndex((item) => item.id === active.id) + 1} 章 / 阅读</span><button onClick={() => setActiveId(chapters[Math.min(chapters.length - 1, chapters.findIndex((item) => item.id === active.id) + 1)].id)} disabled={active.id === chapters[chapters.length - 1].id}>下一章</button></div>
          </>}
        </article>
      </div>
      <footer><span>可随时从顶部“新手教程”重新打开。</span><button className="primary" onClick={onClose}>开始使用</button></footer>
    </section>
  </div>;
}

