export default function App() {
  // Colors
  const bg = "#1e1e24";
  const context = "#7b8cde";    // soft blue-purple for temporal context
  const contextDim = "#7b8cde40";
  const contextGlow = "#7b8cde80";
  const orange = "#e8935a";     // warm orange for surprise/boundary
  const green = "#7aab8e";      // muted green for normal encoding
  const greenDim = "#7aab8e60";
  const textColor = "#e0e0e4";
  const dimText = "#9a9aa8";
  const beltColor = "#3a3a4a";
  const beltStroke = "#55556a";

  return (
    <div className="size-full flex items-center justify-center" style={{ background: bg }}>
      <svg
        viewBox="0 0 1200 760"
        className="w-full h-full max-w-[1200px] max-h-[760px]"
        style={{ fontFamily: "'Inter', 'Helvetica Neue', sans-serif" }}
      >
        <defs>
          {/* Glow filter for current context marker */}
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="6" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <filter id="glowSoft" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          {/* Arrow marker */}
          <marker id="arrowOrange" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6" fill="none" stroke={orange} strokeWidth="1.2" />
          </marker>
          <marker id="arrowContext" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6" fill="none" stroke={context} strokeWidth="1.2" />
          </marker>
          <marker id="arrowGreen" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6" fill="none" stroke={green} strokeWidth="1.2" />
          </marker>
          <marker id="arrowDim" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <path d="M0,0 L8,3 L0,6" fill="none" stroke={dimText} strokeWidth="1.2" />
          </marker>
        </defs>

        {/* ============ TOP ROW ============ */}
        {/* Title */}
        <text x="600" y="42" textAnchor="middle" fill={textColor} fontSize="20" fontWeight="500" letterSpacing="0.3">
          How temporal context organizes memory
        </text>

        {/* === Conveyor Belt === */}
        {(() => {
          const beltY = 240;
          const beltH = 28;
          const beltX = 120;
          const beltW = 960;
          const itemY = 130;
          const itemW = 100;
          const itemH = 40;
          // 4 items evenly spaced
          const items = [
            { x: 260, label: "Item A", state: "normal" },
            { x: 460, label: "Item B", state: "normal" },
            { x: 660, label: "Item C", state: "recalled" },
            { x: 860, label: "Item D", state: "cued" },
          ];
          // Current context position (near item C, slightly right)
          const ctxX = 700;

          return (
            <g>
              {/* Belt track */}
              <rect x={beltX} y={beltY} width={beltW} height={beltH} rx="6" fill={beltColor} stroke={beltStroke} strokeWidth="1" />
              {/* Belt tick marks */}
              {Array.from({ length: 25 }).map((_, i) => (
                <line
                  key={i}
                  x1={beltX + 20 + i * (beltW - 40) / 24}
                  y1={beltY + beltH - 3}
                  x2={beltX + 20 + i * (beltW - 40) / 24}
                  y2={beltY + beltH}
                  stroke={beltStroke}
                  strokeWidth="1"
                  opacity="0.5"
                />
              ))}
              {/* Belt label */}
              <text x={beltX + beltW / 2} y={beltY + beltH + 20} textAnchor="middle" fill={context} fontSize="11" letterSpacing="1.5" opacity="0.7">
                TEMPORAL CONTEXT
              </text>
              {/* Direction arrow on belt */}
              <line x1={beltX + 30} y1={beltY + beltH / 2} x2={beltX + 80} y2={beltY + beltH / 2} stroke={contextDim} strokeWidth="1.5" markerEnd="url(#arrowContext)" />

              {/* Current context marker - glowing */}
              <circle cx={ctxX} cy={beltY + beltH / 2} r="8" fill={context} filter="url(#glow)" opacity="0.9" />
              <circle cx={ctxX} cy={beltY + beltH / 2} r="4" fill="#fff" opacity="0.8" />
              {/* Label for current context */}
              <text x={ctxX} y={beltY + beltH + 38} textAnchor="middle" fill={context} fontSize="10" opacity="0.8">
                current context
              </text>

              {/* Items */}
              {items.map((item, i) => {
                const isRecalled = item.state === "recalled";
                const isCued = item.state === "cued";
                const boxFill = isRecalled ? orange : isCued ? greenDim : green;
                const boxStroke = isRecalled ? orange : isCued ? green : green;
                const boxOpacity = isCued ? 0.5 : 1;
                const textFill = isRecalled ? "#fff" : isCued ? green : "#fff";

                return (
                  <g key={i} opacity={boxOpacity}>
                    {/* Connector line from item to belt */}
                    <line
                      x1={item.x}
                      y1={itemY + itemH}
                      x2={item.x}
                      y2={beltY}
                      stroke={boxStroke}
                      strokeWidth="1"
                      strokeDasharray={isCued ? "4,3" : "none"}
                      opacity={isCued ? 0.5 : 0.4}
                    />
                    {/* Small circle at belt connection */}
                    <circle cx={item.x} cy={beltY + 4} r="3" fill={boxStroke} opacity={isCued ? 0.4 : 0.6} />

                    {/* Item box */}
                    <rect
                      x={item.x - itemW / 2}
                      y={itemY}
                      width={itemW}
                      height={itemH}
                      rx="8"
                      fill={isRecalled ? `${orange}20` : isCued ? `${green}10` : `${green}15`}
                      stroke={boxStroke}
                      strokeWidth={isRecalled ? 1.5 : 1}
                    />
                    <text x={item.x} y={itemY + itemH / 2 + 4} textAnchor="middle" fill={textFill} fontSize="12">
                      {item.label}
                    </text>

                    {/* Recalled item highlight ring */}
                    {isRecalled && (
                      <rect
                        x={item.x - itemW / 2 - 4}
                        y={itemY - 4}
                        width={itemW + 8}
                        height={itemH + 8}
                        rx="10"
                        fill="none"
                        stroke={orange}
                        strokeWidth="1"
                        opacity="0.3"
                        filter="url(#glowSoft)"
                      />
                    )}
                  </g>
                );
              })}

              {/* === Callout annotations === */}

              {/* 1: "item encoded onto context" — small label near Item A connector */}
              <text x={195} y={195} fill={dimText} fontSize="9.5" textAnchor="end">
                item encoded
              </text>
              <text x={195} y={207} fill={dimText} fontSize="9.5" textAnchor="end">
                onto context
              </text>
              <line x1={200} y1={198} x2={240} y2={210} stroke={dimText} strokeWidth="0.7" strokeDasharray="3,2" />

              {/* 2: "current context cues item" — arrow from context marker up to Item C area */}
              <path
                d={`M ${ctxX} ${beltY - 2} Q ${ctxX + 20} ${beltY - 40} ${660} ${itemY + itemH + 6}`}
                fill="none"
                stroke={context}
                strokeWidth="1"
                strokeDasharray="4,3"
                markerEnd="url(#arrowContext)"
                opacity="0.6"
              />
              <text x={ctxX + 50} y={beltY - 35} fill={context} fontSize="9.5" opacity="0.8">
                current context
              </text>
              <text x={ctxX + 50} y={beltY - 23} fill={context} fontSize="9.5" opacity="0.8">
                cues item
              </text>

              {/* 3: "recalled item pulls context" — curved arrow from Item C back down to belt */}
              <path
                d={`M ${660 + 52} ${itemY + itemH / 2} Q ${660 + 100} ${itemY + itemH / 2 + 50} ${ctxX + 5} ${beltY - 5}`}
                fill="none"
                stroke={orange}
                strokeWidth="1.2"
                markerEnd="url(#arrowOrange)"
              />
              <text x={660 + 95} y={itemY + itemH / 2 + 20} fill={orange} fontSize="9.5">
                recalled item
              </text>
              <text x={660 + 95} y={itemY + itemH / 2 + 32} fill={orange} fontSize="9.5">
                pulls context
              </text>

              {/* 4: "nearby item is cued next" — subtle arrow from belt region toward Item D */}
              <path
                d={`M ${ctxX + 15} ${beltY - 2} Q ${ctxX + 60} ${beltY - 50} ${860} ${itemY + itemH + 6}`}
                fill="none"
                stroke={green}
                strokeWidth="1"
                strokeDasharray="4,3"
                markerEnd="url(#arrowGreen)"
                opacity="0.5"
              />
              <text x={860 + 55} y={itemY + 10} fill={green} fontSize="9.5" opacity="0.7">
                nearby item is
              </text>
              <text x={860 + 55} y={itemY + 22} fill={green} fontSize="9.5" opacity="0.7">
                cued next
              </text>
            </g>
          );
        })()}

        {/* Divider */}
        <line x1="160" y1="340" x2="1040" y2="340" stroke={beltStroke} strokeWidth="0.5" opacity="0.5" />

        {/* ============ BOTTOM ROW ============ */}
        <text x="600" y="380" textAnchor="middle" fill={textColor} fontSize="20" fontWeight="500" letterSpacing="0.3">
          What drift rate changes
        </text>

        {/* === Left Panel: Low Drift === */}
        {(() => {
          const px = 160; // panel x
          const py = 410;
          const pw = 400;
          const ph = 300;
          const bY = 520;
          const bH = 22;
          const iY = 470;
          const iW = 70;
          const iH = 32;
          const spacing = 75;
          const startX = px + 80;

          return (
            <g>
              {/* Panel border */}
              <rect x={px} y={py} width={pw} height={ph} rx="10" fill="none" stroke={beltStroke} strokeWidth="0.7" opacity="0.5" />
              {/* Label */}
              <text x={px + pw / 2} y={py + 28} textAnchor="middle" fill={context} fontSize="14" letterSpacing="0.5">
                Low drift
              </text>

              {/* Belt */}
              <rect x={px + 40} y={bY} width={pw - 80} height={bH} rx="5" fill={beltColor} stroke={beltStroke} strokeWidth="0.7" />
              {/* Ticks */}
              {Array.from({ length: 12 }).map((_, i) => (
                <line key={i} x1={px + 60 + i * 26} y1={bY + bH - 2} x2={px + 60 + i * 26} y2={bY + bH} stroke={beltStroke} strokeWidth="0.7" opacity="0.4" />
              ))}

              {/* Items close together */}
              {[0, 1, 2, 3].map((i) => {
                const cx = startX + i * spacing;
                return (
                  <g key={i}>
                    <line x1={cx} y1={iY + iH} x2={cx} y2={bY} stroke={green} strokeWidth="0.8" opacity="0.4" />
                    <circle cx={cx} cy={bY + 3} r="2.5" fill={green} opacity="0.5" />
                    <rect x={cx - iW / 2} y={iY} width={iW} height={iH} rx="6" fill={`${green}15`} stroke={green} strokeWidth="0.8" />
                    <text x={cx} y={iY + iH / 2 + 4} textAnchor="middle" fill={green} fontSize="10" opacity="0.8">
                      Item {i + 1}
                    </text>
                  </g>
                );
              })}

              {/* Spacing brackets */}
              {[0, 1, 2].map((i) => {
                const x1 = startX + i * spacing;
                const x2 = startX + (i + 1) * spacing;
                const bracketY = bY + bH + 16;
                return (
                  <g key={i} opacity="0.35">
                    <line x1={x1} y1={bracketY} x2={x2} y2={bracketY} stroke={context} strokeWidth="0.7" />
                    <line x1={x1} y1={bracketY - 3} x2={x1} y2={bracketY + 3} stroke={context} strokeWidth="0.7" />
                    <line x1={x2} y1={bracketY - 3} x2={x2} y2={bracketY + 3} stroke={context} strokeWidth="0.7" />
                    <text x={(x1 + x2) / 2} y={bracketY + 14} textAnchor="middle" fill={context} fontSize="8">
                      Δc
                    </text>
                  </g>
                );
              })}

              {/* Caption */}
              <text x={px + pw / 2} y={py + ph - 20} textAnchor="middle" fill={dimText} fontSize="11">
                small context shift between items
              </text>
            </g>
          );
        })()}

        {/* === Right Panel: High Drift === */}
        {(() => {
          const px = 640;
          const py = 410;
          const pw = 400;
          const ph = 300;
          const bY = 520;
          const bH = 22;
          const iY = 470;
          const iW = 70;
          const iH = 32;
          const startX = px + 55;
          // Items with varying spacing — large gap after item 2 (surprise)
          const positions = [0, 75, 150, 310]; // last gap is huge

          return (
            <g>
              {/* Panel border */}
              <rect x={px} y={py} width={pw} height={ph} rx="10" fill="none" stroke={beltStroke} strokeWidth="0.7" opacity="0.5" />
              {/* Label */}
              <text x={px + pw / 2} y={py + 28} textAnchor="middle" fill={orange} fontSize="14" letterSpacing="0.5">
                High drift
              </text>

              {/* Belt */}
              <rect x={px + 30} y={bY} width={pw - 60} height={bH} rx="5" fill={beltColor} stroke={beltStroke} strokeWidth="0.7" />
              {Array.from({ length: 14 }).map((_, i) => (
                <line key={i} x1={px + 50 + i * 24} y1={bY + bH - 2} x2={px + 50 + i * 24} y2={bY + bH} stroke={beltStroke} strokeWidth="0.7" opacity="0.4" />
              ))}

              {/* Items */}
              {positions.map((offset, i) => {
                const cx = startX + offset;
                const isSurprise = i === 2;
                const afterBoundary = i === 3;
                const col = isSurprise ? orange : green;
                const bgFill = isSurprise ? `${orange}20` : `${green}15`;
                return (
                  <g key={i}>
                    <line x1={cx} y1={iY + iH} x2={cx} y2={bY} stroke={col} strokeWidth="0.8" opacity="0.4" />
                    <circle cx={cx} cy={bY + 3} r="2.5" fill={col} opacity="0.5" />
                    <rect x={cx - iW / 2} y={iY} width={iW} height={iH} rx="6" fill={bgFill} stroke={col} strokeWidth={isSurprise ? 1.2 : 0.8} />
                    <text x={cx} y={iY + iH / 2 + 4} textAnchor="middle" fill={isSurprise ? orange : green} fontSize="10" opacity={0.8}>
                      {isSurprise ? "Surprise" : afterBoundary ? "Item 4" : `Item ${i + 1}`}
                    </text>
                    {/* Surprise highlight */}
                    {isSurprise && (
                      <rect x={cx - iW / 2 - 3} y={iY - 3} width={iW + 6} height={iH + 6} rx="8" fill="none" stroke={orange} strokeWidth="0.8" opacity="0.3" filter="url(#glowSoft)" />
                    )}
                  </g>
                );
              })}

              {/* Spacing brackets — show small gaps then big gap */}
              {[
                { x1: startX, x2: startX + 75, big: false },
                { x1: startX + 75, x2: startX + 150, big: false },
                { x1: startX + 150, x2: startX + 310, big: true },
              ].map((seg, i) => {
                const bracketY = bY + bH + 16;
                const col = seg.big ? orange : context;
                return (
                  <g key={i} opacity={seg.big ? 0.6 : 0.35}>
                    <line x1={seg.x1} y1={bracketY} x2={seg.x2} y2={bracketY} stroke={col} strokeWidth={seg.big ? 1 : 0.7} />
                    <line x1={seg.x1} y1={bracketY - 3} x2={seg.x1} y2={bracketY + 3} stroke={col} strokeWidth={seg.big ? 1 : 0.7} />
                    <line x1={seg.x2} y1={bracketY - 3} x2={seg.x2} y2={bracketY + 3} stroke={col} strokeWidth={seg.big ? 1 : 0.7} />
                    <text x={(seg.x1 + seg.x2) / 2} y={bracketY + 14} textAnchor="middle" fill={col} fontSize={seg.big ? 9 : 8}>
                      {seg.big ? "large Δc" : "Δc"}
                    </text>
                  </g>
                );
              })}

              {/* Boundary label */}
              <text x={startX + 230} y={bY + bH + 50} textAnchor="middle" fill={orange} fontSize="9" opacity="0.7">
                context boundary
              </text>

              {/* Caption */}
              <text x={px + pw / 2} y={py + ph - 20} textAnchor="middle" fill={dimText} fontSize="11">
                large context shift creates a boundary
              </text>
            </g>
          );
        })()}
      </svg>
    </div>
  );
}
