import React from "react";

export default function App() {
  // Colors — high contrast on white
  const bg = "#ffffff";
  const context = "#3e4b9f";    // deeper blue-purple
  const contextDim = "#4a5abc50";
  const orange = "#d4652a";     // rich warm orange
  const green = "#2a7a4e";      // deeper green
  const greenDim = "#2a7a4e50";
  const textColor = "#1a1a2e";
  const dimText = "#2a7a4e";
  const beltColor = "#e4e4ee";
  const beltStroke = "#4a5abc";

  return (
    <div className="size-full flex items-center justify-center" style={{ background: bg }}>
      <svg
        viewBox="0 0 1100 780"
        className="w-full h-full"
        preserveAspectRatio="xMidYMid meet"
        style={{ fontFamily: "'Inter', 'Helvetica Neue', sans-serif" }}
      >
        <defs>
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="5" result="blur" />
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
          <marker id="arrowOrange" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <path d="M0,0 L10,3.5 L0,7" fill="none" stroke={orange} strokeWidth="1.5" />
          </marker>
          <marker id="arrowContext" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <path d="M0,0 L10,3.5 L0,7" fill="none" stroke={context} strokeWidth="1.5" />
          </marker>
          <marker id="arrowContextFilled" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <path d="M0,0 L10,3.5 L0,7 Z" fill={context} stroke="none" />
          </marker>
          <marker id="arrowGreen" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <path d="M0,0 L10,3.5 L0,7" fill="none" stroke={green} strokeWidth="1.5" />
          </marker>
          <marker id="arrowDim" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
            <path d="M0,0 L10,3.5 L0,7" fill="none" stroke={dimText} strokeWidth="1.5" />
          </marker>
          {/* Gradient for current context band */}
          <linearGradient id="ctxBandGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor={context} stopOpacity="0" />
            <stop offset="15%" stopColor={context} stopOpacity="0.15" />
            <stop offset="40%" stopColor={context} stopOpacity="0.7" />
            <stop offset="50%" stopColor={context} stopOpacity="0.85" />
            <stop offset="60%" stopColor={context} stopOpacity="0.7" />
            <stop offset="85%" stopColor={context} stopOpacity="0.15" />
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

          return (
            <g>
              {/* Belt track */}
              <rect x={beltX} y={beltY} width={beltW} height={beltH} rx="7" fill={beltColor} stroke={beltStroke} strokeWidth="1.5" />
              {/* Belt tick marks */}
              {Array.from({ length: 30 }).map((_, i) => (
                <line
                  key={i}
                  x1={beltX + 18 + i * (beltW - 36) / 29}
                  y1={beltY + beltH - 4}
                  x2={beltX + 18 + i * (beltW - 36) / 29}
                  y2={beltY + beltH}
                  stroke={beltStroke}
                  strokeWidth="1"
                  opacity="0.5"
                />
              ))}
              {/* Belt label */}
              <text x={beltX + beltW / 2} y={beltY + beltH + 24} textAnchor="middle" fill={context} fontSize="13" letterSpacing="2" fontWeight="500">
                TEMPORAL CONTEXT
              </text>
              {/* Direction arrow on belt — removed from start, now placed after band */}

              {/* Current context band — gradient: edges light, center dark */}
              {(() => {
                const bandW = 280;
                const bandX = ctxX - bandW / 2;
                return (
                  <>
                    <rect
                      x={bandX}
                      y={beltY - 4}
                      width={bandW}
                      height={beltH + 8}
                      rx="6"
                      fill="url(#ctxBandGrad)"
                    />
                    {/* Direction arrow overlaid on the band */}
                    <line x1={ctxX - 35} y1={beltY + beltH / 2} x2={ctxX + 35} y2={beltY + beltH / 2} stroke={context} strokeWidth="2" markerEnd="url(#arrowContextFilled)" />
                  </>
                );
              })()}
              {/* Label for current context */}
              <text x={ctxX} y={beltY + beltH + 46} textAnchor="middle" fill={context} fontSize="12" fontWeight="500">
                current context
              </text>

              {/* Items */}
              {items.map((item, i) => {
                const isRecalled = item.state === "recalled";
                const isCued = item.state === "cued";
                const boxStroke = isRecalled ? orange : isCued ? green : green;
                const boxOpacity = isCued ? 0.45 : 1;
                const labelFill = isRecalled ? orange : isCued ? green : textColor;

                return (
                  <g key={i} opacity={boxOpacity}>
                    {/* Connector line */}
                    <line
                      x1={item.x}
                      y1={itemY + itemH}
                      x2={item.x}
                      y2={beltY}
                      stroke={boxStroke}
                      strokeWidth="1.5"
                      strokeDasharray={isCued ? "5,4" : "none"}
                      opacity={isCued ? 0.5 : 0.5}
                    />
                    <circle cx={item.x} cy={beltY + 5} r="4" fill={boxStroke} opacity={isCued ? 0.4 : 0.7} />

                    {/* Item box */}
                    <rect
                      x={item.x - itemW / 2}
                      y={itemY}
                      width={itemW}
                      height={itemH}
                      rx="10"
                      fill={isRecalled ? `${orange}18` : isCued ? `${green}0c` : `${green}12`}
                      stroke={boxStroke}
                      strokeWidth={isRecalled ? 2 : 1.5}
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
              <line x1={153} y1={200} x2={200} y2={215} stroke={dimText} strokeWidth="1" strokeDasharray="4,3" />

              {/* 2: "current context cues item" */}
              <path
                d={`M ${ctxX} ${beltY - 2} Q ${ctxX + 25} ${beltY - 50} ${620} ${itemY + itemH + 8}`}
                fill="none"
                stroke={context}
                strokeWidth="1.5"
                strokeDasharray="5,4"
                markerEnd="url(#arrowContext)"
                opacity="0.7"
              />
              <text x={ctxX + 55} y={beltY - 42} fill={context} fontSize="12" fontWeight="500">
                current context
              </text>
              <text x={ctxX + 55} y={beltY - 28} fill={context} fontSize="12" fontWeight="500">
                cues item
              </text>

              {/* 3: "recalled item pulls context" */}
              <path
                d={`M ${620 + 62} ${itemY + itemH / 2} Q ${620 + 115} ${itemY + itemH / 2 + 55} ${ctxX + 6} ${beltY - 6}`}
                fill="none"
                stroke={orange}
                strokeWidth="1.5"
                markerEnd="url(#arrowOrange)"
              />
              <text x={620 + 108} y={itemY + itemH / 2 + 22} fill={orange} fontSize="12" fontWeight="500">
                recalled item
              </text>
              <text x={620 + 108} y={itemY + itemH / 2 + 37} fill={orange} fontSize="12" fontWeight="500">
                pulls context
              </text>

              {/* 4: "nearby item is cued next" */}
              <path
                d={`M ${ctxX + 18} ${beltY - 2} Q ${ctxX + 75} ${beltY - 55} ${830} ${itemY + itemH + 8}`}
                fill="none"
                stroke={green}
                strokeWidth="1.5"
                strokeDasharray="5,4"
                markerEnd="url(#arrowGreen)"
                opacity="0.6"
              />
              <text x={830 + 65} y={itemY + 8} fill={green} fontSize="12" fontWeight="500">
                nearby item is
              </text>
              <text x={830 + 65} y={itemY + 23} fill={green} fontSize="12" fontWeight="500">
                cued next
              </text>
            </g>
          );
        })()}

        {/* Divider */}
        <line x1="80" y1="360" x2="1020" y2="360" stroke={beltStroke} strokeWidth="0.8" opacity="0.5" />

        {/* ============ BOTTOM ROW ============ */}
        <text x="550" y="400" textAnchor="middle" fill={textColor} fontSize="24" fontWeight="600" letterSpacing="0.2">
          What drift rate changes
        </text>

        {/* === Left Panel: Low Drift === */}
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
          const spacing = 70; // tighter so boxes overlap slightly
          const startX = px + 115;

          return (
            <g>
              <text x={px + pw / 2} y={py + 32} textAnchor="middle" fill={context} fontSize="17" letterSpacing="0.5" fontWeight="600">
                Low drift
              </text>

              {/* Belt */}
              <rect x={px + 40} y={bY} width={pw - 80} height={bH} rx="6" fill={beltColor} stroke={beltStroke} strokeWidth="1" />
              {Array.from({ length: 15 }).map((_, i) => (
                <line key={i} x1={px + 60 + i * 26} y1={bY + bH - 3} x2={px + 60 + i * 26} y2={bY + bH} stroke={beltStroke} strokeWidth="1" opacity="0.4" />
              ))}

              {/* Items */}
              {[0, 1, 2, 3].map((i) => {
                const cx = startX + i * spacing;
                return (
                  <g key={i}>
                    <line x1={cx} y1={iY + iH} x2={cx} y2={bY} stroke={green} strokeWidth="1.2" opacity="0.5" />
                    <circle cx={cx} cy={bY + 4} r="3.5" fill={green} opacity="0.6" />
                    <rect x={cx - iW / 2} y={iY} width={iW} height={iH} rx="8" fill={`${green}12`} stroke={green} strokeWidth="1.2" />
                    <text x={cx} y={iY + iH / 2 + 5} textAnchor="middle" fill={green} fontSize="13" fontWeight="500">
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
                    <line x1={x1} y1={bracketY} x2={x2} y2={bracketY} stroke={context} strokeWidth="1" />
                    <line x1={x1} y1={bracketY - 4} x2={x1} y2={bracketY + 4} stroke={context} strokeWidth="1" />
                    <line x1={x2} y1={bracketY - 4} x2={x2} y2={bracketY + 4} stroke={context} strokeWidth="1" />
                    <text x={(x1 + x2) / 2} y={bracketY + 18} textAnchor="middle" fill={context} fontSize="11" fontWeight="500">
                      Δc
                    </text>
                  </g>
                );
              })}

              <text x={px + pw / 2} y={py + ph - 18} textAnchor="middle" fill={dimText} fontSize="13">
                small context shift between items
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
          const positions = [0, 85, 165, 360];

          return (
            <g>
              <text x={px + pw / 2} y={py + 32} textAnchor="middle" fill={orange} fontSize="17" letterSpacing="0.5" fontWeight="600">
                High drift
              </text>

              {/* Belt */}
              <rect x={px + 25} y={bY} width={pw - 50} height={bH} rx="6" fill={beltColor} stroke={beltStroke} strokeWidth="1" />
              {Array.from({ length: 17 }).map((_, i) => (
                <line key={i} x1={px + 45 + i * 24} y1={bY + bH - 3} x2={px + 45 + i * 24} y2={bY + bH} stroke={beltStroke} strokeWidth="1" opacity="0.4" />
              ))}

              {/* Items */}
              {positions.map((offset, i) => {
                const cx = startX + offset;
                const isSurprise = i === 2;
                const afterBoundary = i === 3;
                const col = isSurprise ? orange : green;
                const bgFill = isSurprise ? `${orange}18` : `${green}12`;
                return (
                  <g key={i}>
                    <line x1={cx} y1={iY + iH} x2={cx} y2={bY} stroke={col} strokeWidth="1.2" opacity="0.5" />
                    <circle cx={cx} cy={bY + 4} r="3.5" fill={col} opacity="0.6" />
                    <rect x={cx - iW / 2} y={iY} width={iW} height={iH} rx="8" fill={bgFill} stroke={col} strokeWidth={isSurprise ? 2 : 1.2} />
                    <text x={cx} y={iY + iH / 2 + 5} textAnchor="middle" fill={isSurprise ? orange : green} fontSize="13" fontWeight="500">
                      {isSurprise ? "Surprise" : afterBoundary ? "Item 4" : `Item ${i + 1}`}
                    </text>
                    {isSurprise && (
                      <rect x={cx - iW / 2 - 4} y={iY - 4} width={iW + 8} height={iH + 8} rx="11" fill="none" stroke={orange} strokeWidth="1.5" opacity="0.4" filter="url(#glowSoft)" />
                    )}
                  </g>
                );
              })}

              {/* Spacing brackets */}
              {[
                { x1: startX, x2: startX + 85, big: false },
                { x1: startX + 85, x2: startX + 165, big: false },
                { x1: startX + 165, x2: startX + 360, big: true },
              ].map((seg, i) => {
                const bracketY = bY + bH + 20;
                const col = seg.big ? orange : context;
                return (
                  <g key={i} opacity={seg.big ? 0.75 : 0.5}>
                    <line x1={seg.x1} y1={bracketY} x2={seg.x2} y2={bracketY} stroke={col} strokeWidth={seg.big ? 1.5 : 1} />
                    <line x1={seg.x1} y1={bracketY - 4} x2={seg.x1} y2={bracketY + 4} stroke={col} strokeWidth={seg.big ? 1.5 : 1} />
                    <line x1={seg.x2} y1={bracketY - 4} x2={seg.x2} y2={bracketY + 4} stroke={col} strokeWidth={seg.big ? 1.5 : 1} />
                    <text x={(seg.x1 + seg.x2) / 2} y={bracketY + 18} textAnchor="middle" fill={col} fontSize={seg.big ? 12 : 11} fontWeight="500">
                      {seg.big ? "large Δc" : "Δc"}
                    </text>
                  </g>
                );
              })}

              {/* Boundary label */}
              <text x={startX + 262} y={bY + bH + 58} textAnchor="middle" fill={orange} fontSize="12" fontWeight="500">
                context boundary
              </text>

              <text x={px + pw / 2} y={py + ph - 18} textAnchor="middle" fill={dimText} fontSize="13">
                large context shift creates a boundary
              </text>
            </g>
          );
        })()}
      </svg>
    </div>
  );
}