export default function App() {
  // Colors
  const bg = "#ffffff";
  const context = "#555570";
  const orange = "#d4652a";
  const textColor = "#555570";
  const dimText = "#555570";
  const beltStroke = "#8888a0";

  // Cividis colormap sampled stops for interpolation
  const cividisStops = [
    { t: 0.0, r: 0, g: 32, b: 77 },
    { t: 0.1, r: 0, g: 42, b: 100 },
    { t: 0.2, r: 27, g: 59, b: 111 },
    { t: 0.3, r: 61, g: 78, b: 122 },
    { t: 0.4, r: 91, g: 97, b: 129 },
    { t: 0.5, r: 122, g: 123, b: 120 },
    { t: 0.6, r: 154, g: 148, b: 98 },
    { t: 0.7, r: 184, g: 176, b: 64 },
    { t: 0.8, r: 207, g: 199, b: 49 },
    { t: 0.9, r: 230, g: 223, b: 44 },
    { t: 1.0, r: 254, g: 232, b: 56 },
  ];

  function sampleCividis(t: number): string {
    const clamped = Math.max(0, Math.min(1, t));
    let i = 0;
    for (let j = 0; j < cividisStops.length - 1; j++) {
      if (cividisStops[j + 1].t >= clamped) { i = j; break; }
    }
    const a = cividisStops[i];
    const b = cividisStops[i + 1];
    const f = (clamped - a.t) / (b.t - a.t);
    const r = Math.round(a.r + (b.r - a.r) * f);
    const g = Math.round(a.g + (b.g - a.g) * f);
    const bl = Math.round(a.b + (b.b - a.b) * f);
    return `rgb(${r},${g},${bl})`;
  }

  // Determine if text should be light or dark on a cividis background
  function cividisTextColor(t: number): string {
    return t < 0.6 ? "#e0e0e8" : "#1a1a2e";
  }

  return (
    <div className="size-full flex items-center justify-center" style={{ background: bg }}>
      <svg
        viewBox="0 0 1100 780"
        className="w-full h-full"
        preserveAspectRatio="xMidYMid meet"
        style={{ fontFamily: "'Inter', 'Helvetica Neue', sans-serif" }}
      >
        <defs>
          <filter id="glowSoft" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <marker id="arrowOrange" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <path d="M0,0 L10,3.5 L0,7 Z" fill={orange} stroke="none" />
          </marker>
          <marker id="arrowContext" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <path d="M0,0 L10,3.5 L0,7 Z" fill={context} stroke="none" />
          </marker>
          <marker id="arrowGreen" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <path d="M0,0 L10,3.5 L0,7 Z" fill="#cfc731" stroke="none" />
          </marker>
          <marker id="arrowDim" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <path d="M0,0 L10,3.5 L0,7 Z" fill={dimText} stroke="none" />
          </marker>

          {/* Cividis gradient for main belt */}
          <linearGradient id="cividisMain" x1="0" y1="0" x2="1" y2="0">
            {cividisStops.map((s, i) => (
              <stop key={i} offset={`${s.t * 100}%`} stopColor={`rgb(${s.r},${s.g},${s.b})`} />
            ))}
          </linearGradient>

          {/* Cividis gradient for low drift belt (narrower range since items are close) */}
          <linearGradient id="cividisLow" x1="0" y1="0" x2="1" y2="0">
            {cividisStops.map((s, i) => (
              <stop key={i} offset={`${s.t * 100}%`} stopColor={`rgb(${s.r},${s.g},${s.b})`} />
            ))}
          </linearGradient>

          {/* Cividis gradient for high drift belt */}
          <linearGradient id="cividisHigh" x1="0" y1="0" x2="1" y2="0">
            {cividisStops.map((s, i) => (
              <stop key={i} offset={`${s.t * 100}%`} stopColor={`rgb(${s.r},${s.g},${s.b})`} />
            ))}
          </linearGradient>

          {/* Current context band overlay — solid blue-purple, fading at edges */}
          <linearGradient id="ctxBandGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor={context} stopOpacity="0" />
            <stop offset="12%" stopColor={context} stopOpacity="0.12" />
            <stop offset="35%" stopColor={context} stopOpacity="0.55" />
            <stop offset="50%" stopColor={context} stopOpacity="0.72" />
            <stop offset="65%" stopColor={context} stopOpacity="0.55" />
            <stop offset="88%" stopColor={context} stopOpacity="0.12" />
            <stop offset="100%" stopColor={context} stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* ============ TOP ROW ============ */}
        <text x="550" y="40" textAnchor="middle" fill={textColor} fontSize="24" fontWeight="600" letterSpacing="0.2">
          How temporal context organizes memory
        </text>

        {/* === Conveyor Belt === */}
        {(() => {
          const beltY = 250;
          const beltH = 32;
          const beltX = 60;
          const beltW = 980;
          const itemY = 120;
          const itemW = 120;
          const itemH = 48;
          const items = [
            { x: 220, label: "Item A", state: "normal" },
            { x: 420, label: "Item B", state: "normal" },
            { x: 620, label: "Item C", state: "recalled" },
            { x: 830, label: "Item D", state: "cued" },
          ];
          const ctxX = 665;
          const bandW = 280;

          // Helper: get cividis t for an x position on this belt
          const beltT = (x: number) => (x - beltX) / beltW;

          return (
            <g>
              {/* Belt track with cividis */}
              <rect x={beltX} y={beltY} width={beltW} height={beltH} rx="7" fill="url(#cividisMain)" stroke={beltStroke} strokeWidth="1" opacity="0.85" />
              {/* Belt tick marks */}
              {Array.from({ length: 30 }).map((_, i) => (
                <line
                  key={i}
                  x1={beltX + 18 + i * (beltW - 36) / 29}
                  y1={beltY + beltH - 4}
                  x2={beltX + 18 + i * (beltW - 36) / 29}
                  y2={beltY + beltH}
                  stroke="#ffffff"
                  strokeWidth="1"
                  opacity="0.2"
                />
              ))}
              {/* Belt label */}
              <text x={beltX + beltW / 2} y={beltY + beltH + 24} textAnchor="middle" fill={dimText} fontSize="13" letterSpacing="2" fontWeight="500">
                TEMPORAL CONTEXT
              </text>

              {/* Current context band — solid hue overlay */}
              {(() => {
                const bandX = ctxX - bandW / 2;
                return (
                  <rect
                    x={bandX}
                    y={beltY - 10}
                    width={bandW}
                    height={beltH + 20}
                    rx="7"
                    fill="url(#ctxBandGrad)"
                  />
                );
              })()}
              {/* Direction arrow — parallel below the band */}
              {/* Label for current context */}
              <text x={ctxX} y={beltY + beltH + 48} textAnchor="middle" fill={context} fontSize="12" fontWeight="500">
                current context "drifting"
              </text>
              <line x1={ctxX - 40} y1={beltY + beltH + 60} x2={ctxX + 40} y2={beltY + beltH + 60} stroke={context} strokeWidth="2" markerEnd="url(#arrowContext)" />

              {/* Items — colored to match belt position */}
              {items.map((item, i) => {
                const isRecalled = item.state === "recalled";
                const isCued = item.state === "cued";
                const t = beltT(item.x);
                const itemColor = isRecalled ? orange : sampleCividis(t);
                const boxOpacity = isCued ? 0.4 : 1;
                const labelFill = isRecalled ? orange : cividisTextColor(t);

                return (
                  <g key={i} opacity={boxOpacity}>
                    {/* Connector line */}
                    <line
                      x1={item.x}
                      y1={itemY + itemH}
                      x2={item.x}
                      y2={beltY}
                      stroke={itemColor}
                      strokeWidth="1.5"
                      strokeDasharray={isCued ? "5,4" : "none"}
                      opacity={0.5}
                    />
                    <circle cx={item.x} cy={beltY + 5} r="4" fill={itemColor} opacity={0.7} />

                    {/* Item box */}
                    <rect
                      x={item.x - itemW / 2}
                      y={itemY}
                      width={itemW}
                      height={itemH}
                      rx="10"
                      fill={isRecalled ? `${orange}20` : sampleCividis(t)}
                      stroke={isRecalled ? orange : sampleCividis(Math.max(0, t - 0.05))}
                      strokeWidth={isRecalled ? 2 : 1.5}
                      opacity={isRecalled ? 1 : 0.85}
                    />
                    <text x={item.x} y={itemY + itemH / 2 + 5} textAnchor="middle" fill={labelFill} fontSize="14" fontWeight="500">
                      {item.label}
                    </text>

                    {/* Recalled highlight ring */}
                    {isRecalled && (
                      <rect
                        x={item.x - itemW / 2 - 5}
                        y={itemY - 5}
                        width={itemW + 10}
                        height={itemH + 10}
                        rx="13"
                        fill="none"
                        stroke={orange}
                        strokeWidth="1.5"
                        opacity="0.35"
                        filter="url(#glowSoft)"
                      />
                    )}
                  </g>
                );
              })}

              {/* === Callout annotations === */}

              {/* 1: "item encoded onto context" */}
              <text x={148} y={195} fill={dimText} fontSize="12" textAnchor="end" fontWeight="500">
                item encoded
              </text>
              <text x={148} y={210} fill={dimText} fontSize="12" textAnchor="end" fontWeight="500">
                onto context
              </text>
              <line x1={153} y1={200} x2={200} y2={215} stroke={dimText} strokeWidth="1" />

              {/* 2: "current context cues item" */}
              <path
                d={`M ${ctxX - 105} ${beltY - 5} Q ${ctxX - 150} ${beltY - 10} ${620 - 70} ${itemY + itemH + 5}`}
                fill="none"
                stroke={context}
                strokeWidth="1.5"
                markerEnd="url(#arrowContext)"
                opacity="0.7"
              />
              <text x={ctxX - 220} y={beltY - 52} fill={context} fontSize="12" fontWeight="500" textAnchor="start">
                current context
              </text>
              <text x={ctxX - 200} y={beltY - 38} fill={context} fontSize="12" fontWeight="500" textAnchor="start">
                cues item
              </text>

              {/* 3: "recalled item pulls context" */}
              <path
                d={`M ${620 + 62} ${itemY + itemH / 2} Q ${620 + 95} ${itemY + itemH / 2 + 45} ${ctxX + 6} ${beltY - 20}`}
                fill="none"
                stroke={orange}
                strokeWidth="1.5"
                markerEnd="url(#arrowOrange)"
              />
              <text x={620 + 80} y={itemY + itemH / 2 + 23} fill={orange} fontSize="12" fontWeight="500">
                recalled item
              </text>
              <text x={620 + 88} y={itemY + itemH / 2 + 37} fill={orange} fontSize="12" fontWeight="500">
                pulls context
              </text>

              {/* 4: "nearby item is cued next" */}
              <path
                d={`M ${ctxX + 18} ${beltY - 20} Q ${ctxX + 75} ${beltY - 65} ${830} ${itemY + itemH + 8}`}
                fill="none"
                stroke="#cfc731"
                strokeWidth="1.5"
                markerEnd="url(#arrowGreen)"
                opacity="0.8"
              />
              <text x={830 + 65} y={itemY + 8} fill="#cfc731" fontSize="12" fontWeight="500">
                nearby item is
              </text>
              <text x={830 + 65} y={itemY + 23} fill="#cfc731" fontSize="12" fontWeight="500">
                cued next
              </text>
            </g>
          );
        })()}

        {/* Divider */}
        <line x1="80" y1="360" x2="1020" y2="360" stroke={beltStroke} strokeWidth="0.8" opacity="0.3" />

        {/* ============ BOTTOM ROW ============ */}
        <text x="550" y="400" textAnchor="middle" fill={textColor} fontSize="24" fontWeight="600" letterSpacing="0.2">
          What drift rate changes
        </text>

        {/* === Left Panel: Normal Drift === */}
        {(() => {
          const px = 60;
          const py = 425;
          const pw = 470;
          const ph = 330;
          const bY = 560;
          const bH = 28;
          const iY = 495;
          const iW = 90;
          const iH = 40;
          const spacing = 70;
          const startX = px + 115;
          const lBeltX = px + 40;
          const lBeltW = pw - 80;

          const beltT = (x: number) => (x - lBeltX) / lBeltW;

          return (
            <g>
              <text x={px + pw / 2} y={py + 32} textAnchor="middle" fill={context} fontSize="17" letterSpacing="0.5" fontWeight="600">
                Normal drift
              </text>

              {/* Belt with cividis */}
              <rect x={lBeltX} y={bY} width={lBeltW} height={bH} rx="6" fill="url(#cividisLow)" stroke={beltStroke} strokeWidth="0.8" opacity="0.85" />
              {Array.from({ length: 15 }).map((_, i) => (
                <line key={i} x1={px + 60 + i * 26} y1={bY + bH - 3} x2={px + 60 + i * 26} y2={bY + bH} stroke="#ffffff" strokeWidth="0.8" opacity="0.15" />
              ))}

              {/* Items — colored to match belt position */}
              {[0, 1, 2, 3].map((i) => {
                const cx = startX + i * spacing;
                const t = beltT(cx);
                const col = sampleCividis(t);
                const txtCol = cividisTextColor(t);
                return (
                  <g key={i}>
                    <line x1={cx} y1={iY + iH} x2={cx} y2={bY} stroke={col} strokeWidth="1.2" opacity="0.5" />
                    <circle cx={cx} cy={bY + 4} r="3.5" fill={col} opacity="0.7" />
                    <rect x={cx - iW / 2} y={iY} width={iW} height={iH} rx="8" fill={col} stroke={sampleCividis(Math.max(0, t - 0.05))} strokeWidth="1.2" opacity="0.85" />
                    <text x={cx} y={iY + iH / 2 + 5} textAnchor="middle" fill={txtCol} fontSize="13" fontWeight="500">
                      Item {i + 1}
                    </text>
                  </g>
                );
              })}

              {/* Spacing brackets */}
              {[0, 1, 2].map((i) => {
                const x1 = startX + i * spacing;
                const x2 = startX + (i + 1) * spacing;
                const bracketY = bY + bH + 20;
                return (
                  <g key={i} opacity="0.5">
                    <line x1={x1} y1={bracketY} x2={x2} y2={bracketY} stroke={dimText} strokeWidth="1" />
                    <line x1={x1} y1={bracketY - 4} x2={x1} y2={bracketY + 4} stroke={dimText} strokeWidth="1" />
                    <line x1={x2} y1={bracketY - 4} x2={x2} y2={bracketY + 4} stroke={dimText} strokeWidth="1" />
                    <text x={(x1 + x2) / 2} y={bracketY + 18} textAnchor="middle" fill={dimText} fontSize="11" fontWeight="500">
                      Δc
                    </text>
                  </g>
                );
              })}

              <text x={px + pw / 2} y={py + ph - 100} textAnchor="middle" fill={dimText} fontSize="13" fontWeight="500">
                Gradually drifting context across items
              </text>
            </g>
          );
        })()}

        {/* === Right Panel: High Drift === */}
        {(() => {
          const px = 570;
          const py = 425;
          const pw = 470;
          const ph = 330;
          const bY = 560;
          const bH = 28;
          const iY = 495;
          const iW = 90;
          const iH = 40;
          const startX = px + 55;
          const positions = [0, 65, 125, 271];
          const hBeltX = px + 25;
          const hBeltW = pw - 50;

          const beltT = (x: number) => (x - hBeltX) / hBeltW;

          return (
            <g>
              <text x={px + pw / 2} y={py + 32} textAnchor="middle" fill={orange} fontSize="17" letterSpacing="0.5" fontWeight="600">
                High drift
              </text>

              {/* Belt with cividis */}
              <rect x={hBeltX} y={bY} width={hBeltW} height={bH} rx="6" fill="url(#cividisHigh)" stroke={beltStroke} strokeWidth="0.8" opacity="0.85" />
              {Array.from({ length: 17 }).map((_, i) => (
                <line key={i} x1={px + 45 + i * 24} y1={bY + bH - 3} x2={px + 45 + i * 24} y2={bY + bH} stroke="#ffffff" strokeWidth="0.8" opacity="0.15" />
              ))}

              {/* Items — colored to match belt, surprise in orange */}
              {positions.map((offset, i) => {
                const cx = startX + offset;
                const isSurprise = i === 2;
                const afterBoundary = i === 3;
                const t = beltT(cx);
                const col = isSurprise ? orange : sampleCividis(t);
                const bgFill = isSurprise ? `${orange}20` : sampleCividis(t);
                const txtCol = isSurprise ? orange : cividisTextColor(t);
                const strokeCol = isSurprise ? orange : sampleCividis(Math.max(0, t - 0.05));
                return (
                  <g key={i}>
                    <line x1={cx} y1={iY + iH} x2={cx} y2={bY} stroke={col} strokeWidth="1.2" opacity="0.5" />
                    <circle cx={cx} cy={bY + 4} r="3.5" fill={col} opacity="0.7" />
                    <rect
                      x={cx - iW / 2} y={iY} width={iW} height={iH} rx="8"
                      fill={bgFill}
                      stroke={strokeCol}
                      strokeWidth={isSurprise ? 2 : 1.2}
                      opacity={isSurprise ? 1 : 0.85}
                    />
                    <text x={cx} y={iY + iH / 2 + 5} textAnchor="middle" fill={txtCol} fontSize="13" fontWeight="500">
                      {isSurprise ? "Surprise" : afterBoundary ? "Item 4" : `Item ${i + 1}`}
                    </text>
                    {isSurprise && (
                      <rect x={cx - iW / 2 - 4} y={iY - 4} width={iW + 8} height={iH + 8} rx="11" fill="none" stroke={orange} strokeWidth="1.5" opacity="0.5" filter="url(#glowSoft)" fontWeight="500"/>
                    )}
                  </g>
                );
              })}

              {/* Spacing brackets */}
              {[
                { x1: startX, x2: startX + 65, big: false },
                { x1: startX + 65, x2: startX + 125, big: false },
                { x1: startX + 125, x2: startX + 271, big: true },
              ].map((seg, i) => {
                const bracketY = bY + bH + 20;
                const col = seg.big ? orange : dimText;
                return (
                  <g key={i} opacity={seg.big ? 0.75 : 0.5}>
                    <line x1={seg.x1} y1={bracketY} x2={seg.x2} y2={bracketY} stroke={col} strokeWidth={seg.big ? 1.5 : 1} />
                    <line x1={seg.x1} y1={bracketY - 4} x2={seg.x1} y2={bracketY + 4} stroke={col} strokeWidth={seg.big ? 1.5 : 1} />
                    <line x1={seg.x2} y1={bracketY - 4} x2={seg.x2} y2={bracketY + 4} stroke={col} strokeWidth={seg.big ? 1.5 : 1} />
                    <text x={(seg.x1 + seg.x2) / 2} y={bracketY + 18} textAnchor="middle" fill={col} fontSize={seg.big ? 12 : 11} fontWeight="500">
                      {seg.big ? "reduced bry Δc" : "reduced Δc"}
                    </text>
                  </g>
                );
              })}

              <text x={px + pw / 2} y={py + ph - 100} textAnchor="middle" fill={dimText} fontSize="13" fontWeight="500">
                Faster drifting context causes discontinuity → "boundary"
              </text>
            </g>
          );
        })()}
      </svg>
    </div>
  );
}