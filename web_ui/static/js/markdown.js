/* Minimal markdown renderer with syntax highlighting.
 *
 * Self-contained on purpose: no CDN, no build step, works offline. Everything
 * is HTML-escaped before any markup is generated, so model output can never
 * inject nodes into the page.
 */
(function (global) {
  'use strict';

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  /* Syntax highlighting via highlight.js (vendored, loaded in base.html).
   * Takes RAW code and returns escaped, token-wrapped HTML. Falls back to plain
   * escaped text when the language is unknown or the blob is too big. */
  const MAX_HIGHLIGHT_CHARS = 40000;

  function highlight(code, lang) {
    if (!code) return '';
    const key = (lang || '').toLowerCase();
    if (key && code.length <= MAX_HIGHLIGHT_CHARS && global.hljs) {
      try {
        return global.hljs.highlight(code, { language: key, ignoreIllegals: true }).value;
      } catch (_) { /* unknown language: fall through to plain text */ }
    }
    return escapeHtml(code);
  }

  /* Cut highlighted HTML into one string per line, reopening on each line
   * whatever spans were still open at the break.
   *
   * The tokens hljs emits are `<span class=...>`, `</span>` and escaped text,
   * and nothing else -- escaped text can contain no raw `<` -- so this is a
   * complete grammar for its output rather than an attempt at parsing HTML.
   */
  function splitHighlighted(html) {
    const lines = [];
    const open = [];
    let current = '';
    const token = /<span\b[^>]*>|<\/span>|\n|[^<\n]+|</g;
    let m;
    while ((m = token.exec(html)) !== null) {
      const tok = m[0];
      if (tok === '\n') {
        lines.push(current + '</span>'.repeat(open.length));
        current = open.join('');
      } else if (tok === '</span>') {
        open.pop();
        current += tok;
      } else if (tok.charCodeAt(0) === 60 && tok.charCodeAt(1) !== 47 && tok.length > 1) {
        open.push(tok);
        current += tok;
      } else {
        current += tok;
      }
    }
    lines.push(current + '</span>'.repeat(open.length));
    return lines;
  }

  /* Highlight a whole block, then hand back its lines.
   *
   * This exists because the line-numbered views used to call `highlight` once
   * per line, and a highlighter has no memory between calls. Two things broke
   * that way, both reported from real files:
   *
   *   - a block comment only coloured its first line, because the lines after
   *     it never saw the `/*` that opened it;
   *   - `<script>` inside an HTML file was not highlighted as JavaScript at
   *     all. hljs hands the body of a script tag to its javascript grammar as
   *     a sub-language, and it can only do that when it is given the tag and
   *     its contents together.
   */
  function highlightLines(code, lang) {
    const text = String(code == null ? '' : code);
    const key = (lang || '').toLowerCase();
    if (key && text.length <= MAX_HIGHLIGHT_CHARS && global.hljs) {
      try {
        return splitHighlighted(
          global.hljs.highlight(text, { language: key, ignoreIllegals: true }).value);
      } catch (_) { /* unknown language: fall through to plain text */ }
    }
    return text.split('\n').map(escapeHtml);
  }

  /* A path starts with /, ~/, ./, ../, or a directory segment, then runs on.
   *
   * It used to run to the next whitespace, which breaks every path containing a
   * space: a drive called "Gaming Beast" produced two links, one ending at
   * "Gaming" and another starting at "Beast/". Directory names with spaces in
   * them are entirely normal on removable media and on macOS.
   *
   * So a space is allowed *inside* a path, but only where the path visibly
   * carries on past it -- the run after the space has to reach a "/" or end in
   * a file extension. That is what separates "…/Gaming Beast/data.txt" from
   * "…/env and then stop", where the word after the space is just prose and the
   * path ends. Each branch consumes exactly one character, so there is no
   * nested quantifier here to backtrack over.
   *
   * The negative lookbehind stops the pass from re-linkifying a href value the
   * link pass just wrote. Trailing sentence punctuation is split back out so it
   * is not swallowed by the link. The line (and optional range) ride in data
   * attributes the app reads on click. */
  const PATH_CHAR = '[^\\s<>"\'`]';
  // Rooted: /x, ~/x, ./x, ../x. Only these may contain spaces -- "n/a and/or
  // AC/DC" is a bare segment away from being a path, and allowing spaces there
  // swallowed the lot as one link.
  const ROOTED = '(?:\\/|~\\/|\\.{1,2}\\/)';
  const BARE = '[\\w@.~\\-]+\\/';
  // ...and not across sentence punctuation, or "/tmp/one.txt, /tmp/two.txt"
  // becomes a single path with a comma in the middle of it.
  const SPACE_INSIDE = `(?<![,;:!?)\\]}"'\`]) ${''
    }(?=${PATH_CHAR}*(?:\\/|\\.[A-Za-z0-9]{1,6}(?!${PATH_CHAR})))`;
  /* A bare path directly after a word that itself contains a "/" is a piece of
     something longer, not a path of its own. "`.../encounter tables/extracted/`"
     is an agent writing an abbreviated path; matching from "tables/" gives a
     link to a directory that has never existed, and clicking it says "file not
     found". A truncated link is worse than no link, so there is none.

     Rooted paths are exempt: "/tmp/one.txt, /tmp/two.txt" is two real paths,
     and a leading "/" says so without needing context. */
  const NOT_A_FRAGMENT = '(?<!\\/\\S*\\s)';

  // ...and one last case the "does it carry on" rule cannot see: a path whose
  // final directory has a space in it and which ends the line, like
  // "…/AI-Fantasy-Images/encounter tables". Nothing follows to prove the path
  // continues, so it is taken only when the path already contains a space --
  // otherwise "open /tmp/x done" would swallow "done".
  // The tail must not itself be the start of a path, or "/tmp/one.txt,
  // /tmp/two.txt" consumes the second one as the first one's last segment and
  // it never gets scanned on its own.
  const TRAILING_SEGMENT =
    `(?:( (?!${ROOTED})${PATH_CHAR}+)(?=[.,;:!?)\\]}"'\`]*\\s*$))?`;
  const FILE_REF_SOURCE =
    '(?<![=">])(^|[\\s(["\'`])'
    + `(${ROOTED}(?:${PATH_CHAR}|${SPACE_INSIDE})+|${NOT_A_FRAGMENT}${BARE}${PATH_CHAR}+)`
    + TRAILING_SEGMENT;

  /* A fresh regex per pass. `fileRefReplacer` re-runs this on any text it
     decided not to link, and a global regex carries `lastIndex` on the object
     itself -- sharing one between an outer replace and a nested one corrupts
     the outer iteration. */
  function linkPaths(text) {
    return text.replace(new RegExp(FILE_REF_SOURCE, 'gm'), fileRefReplacer);
  }

  /* "n/a", "and/or", "AC/DC" are prose, not paths. A path is absolute or
   * explicitly relative, nested, or ends in a filename extension.
   *
   * "Nested" alone was too generous. A slash is also how people write a short
   * list -- "col1/2/3", "8345/8347/8352", "L/F/R" -- and each of those has two
   * slashes, so each became a link to a file that has never existed. What
   * separates them from `agent_server/routes` is their *segments*: a real
   * directory name is a word, and these are bare digits and single letters.
   *
   * So a bare nested path with nothing else to recommend it has to be made of
   * name-like parts. A filename extension or a trailing slash still speaks for
   * itself and skips the check, which keeps `dist/1/index.html` and
   * `tables/extracted/`. Rooted paths are never asked: a leading "/" is a claim
   * in itself, and "/1/2/3" is a perfectly good path.
   *
   * Nothing here touches the filesystem. Checking would be exact and would mean
   * a round trip per candidate on every message rendered, which is not a trade
   * worth making for a link that is occasionally wrong. */
  function looksLikePath(p) {
    if (/^(\/|~\/|\.{1,2}\/)/.test(p)) return true;
    if (/\.[A-Za-z0-9]{1,6}$/.test(p)) return true;
    if (p.endsWith('/')) return true;
    const parts = p.split('/').filter(Boolean);
    if (parts.length < 2) return false;
    // A name: at least two characters, and at least one of them a letter.
    return parts.every((part) => part.length >= 2 && /[A-Za-z]/.test(part));
  }

  function fileRefReplacer(full, pre, tok, tail) {
    tail = tail || '';
    /* The trailing segment is deliberately *not* re-scanned when it goes
       unused. It was consumed as a possible last part of this path, so it is
       either the prose after the path or a piece of the same abbreviated path
       -- and linkifying a piece produces "tables/extracted/" out of
       "`.../encounter tables/extracted/`", a link to a directory that has never
       existed. Rooted paths are excluded from being taken as a tail at all, so
       nothing real is lost here. */
    if (/^(https?:\/\/|www\.|mailto:)/i.test(tok)) return full;
    // See TRAILING_SEGMENT. Taken only when this is already a spaced path;
    // otherwise it is the next word of the sentence and goes back untouched.
    if (tail && tok.includes(' ')) { tok += tail; tail = ''; }
    const trail = (tok.match(/[.,;:!?)\]}"'`]+$/) || [''])[0];
    const core = trail ? tok.slice(0, -trail.length) : tok;
    const m = /^(.+?)(?::(\d+)(?:-(\d+))?)?$/.exec(core);
    const path = m ? m[1] : '';
    if (!path || !looksLikePath(path)) return full;
    const line = m[2], end = m[3];
    const display = path + (line ? ':' + line + (end ? '-' + end : '') : '');
    let attrs = 'data-path="' + path + '"';
    if (line) attrs += ' data-line="' + line + '"';
    if (end) attrs += ' data-line-end="' + end + '"';
    return pre + '<a class="file-ref" href="#" ' + attrs + '>' + display + '</a>'
      + trail + tail;
  }

  function inline(text) {
    let out = escapeHtml(text);
    // Inline code first so its contents are not treated as markup.
    const codes = [];
    out = out.replace(/`([^`\n]+)`/g, (_, code) => {
      codes.push(code);
      return '\u0000CODE' + (codes.length - 1) + '\u0000';
    });

    out = out
      .replace(/!\[([^\]]*)\]\(([^)\s]+)[^)]*\)/g,
        (_, alt, src) => `<img src="${src}" alt="${alt}" loading="lazy">`)
      .replace(/\[([^\]]+)\]\(([^)\s]+)[^)]*\)/g,
        (_, label, href) => /^(https?:|\/|#)/.test(href)
          ? `<a href="${href}" target="_blank" rel="noopener noreferrer">${label}</a>`
          : label);

    // Bare URLs become links. The lookbehind skips anything already inside a
    // tag we just generated: after escapeHtml, a literal " or > can only have
    // come from our own markup. Stashed so the emphasis pass below cannot eat
    // underscores inside a URL.
    const links = [];
    out = out.replace(/(?<!["=>])\bhttps?:\/\/[^\s<>"'`]+/g, (url) => {
      // Sentence punctuation and trailing emphasis markers are almost never
      // part of the URL. Only a trailing run is stripped, so underscores and
      // asterisks *inside* a path survive.
      const tail = (url.match(/[.,;:!?)\]}*_~]+$/) || [''])[0];
      const href = tail ? url.slice(0, -tail.length) : url;
      if (!/^https?:\/\/[^/]/.test(href)) return url;
      links.push(`<a href="${href}" target="_blank" rel="noopener noreferrer">${href}</a>`);
      return '\u0000LINK' + (links.length - 1) + '\u0000' + tail;
    });

    // File paths (absolute or relative) and file:line references become links
    // that open the in-app editor. URLs are stashed above, so anything left
    // starting with / or a dir/ segment is a path, not a link.
    out = linkPaths(out
      .replace(/(\*\*|__)(?=\S)([\s\S]*?\S)\1/g, '<strong>$2</strong>')
      .replace(/(^|[\s(])(\*|_)(?=\S)([^*_]*?\S)\2/g, '$1<em>$3</em>')
      .replace(/~~(?=\S)([\s\S]*?\S)~~/g, '<del>$1</del>'));

    return out
      .replace(/\u0000LINK(\d+)\u0000/g, (_, i) => links[+i])
      .replace(/\u0000CODE(\d+)\u0000/g,
        (_, i) => '<code>' +
          linkPaths(codes[+i]) +
          '</code>');
  }

  function render(src) {
    if (!src) return '';
    const lines = String(src).replace(/\r\n?/g, '\n').split('\n');
    const html = [];
    let i = 0;

    const listStack = [];
    function closeLists(toDepth) {
      while (listStack.length > toDepth) html.push(listStack.pop() === 'ol' ? '</ol>' : '</ul>');
    }

    while (i < lines.length) {
      const line = lines[i];

      // Fenced code block
      const fence = line.match(/^\s*(`{3,}|~{3,})\s*([\w+#.-]*)\s*$/);
      if (fence) {
        closeLists(0);
        const marker = fence[1][0];
        const lang = fence[2] || '';
        const body = [];
        i++;
        while (i < lines.length && !new RegExp('^\\s*' + marker + '{3,}\\s*$').test(lines[i])) {
          body.push(lines[i]);
          i++;
        }
        i++;
        const code = body.join('\n');
        html.push(
          '<div class="code-block" data-code="' + escapeHtml(code) + '">' +
          '<div class="code-head"><span class="code-lang">' + escapeHtml(lang || 'text') + '</span>' +
          '<button type="button" class="code-copy" onclick="copyCode(this)" title="Copy to clipboard">' +
          '<svg viewBox="0 0 24 24" width="13" height="13" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" fill="none" stroke="currentColor" stroke-width="2"/></svg>' +
          '</button></div>' +
          '<pre><code>' + highlight(code, lang) + '</code></pre></div>'
        );
        continue;
      }

      // Heading
      const heading = line.match(/^(#{1,6})\s+(.*)$/);
      if (heading) {
        closeLists(0);
        const level = heading[1].length;
        html.push(`<h${level}>${inline(heading[2])}</h${level}>`);
        i++;
        continue;
      }

      // Horizontal rule
      if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) {
        closeLists(0);
        html.push('<hr>');
        i++;
        continue;
      }

      // Blockquote
      if (/^\s*>\s?/.test(line)) {
        closeLists(0);
        const quote = [];
        while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
          quote.push(lines[i].replace(/^\s*>\s?/, ''));
          i++;
        }
        html.push('<blockquote>' + render(quote.join('\n')) + '</blockquote>');
        continue;
      }

      // Table
      if (/\|/.test(line) && i + 1 < lines.length && /^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(lines[i + 1])) {
        closeLists(0);
        const cells = (row) => row.replace(/^\s*\|/, '').replace(/\|\s*$/, '').split('|');
        html.push('<table><thead><tr>' +
          cells(line).map((c) => `<th>${inline(c.trim())}</th>`).join('') +
          '</tr></thead><tbody>');
        i += 2;
        while (i < lines.length && /\|/.test(lines[i]) && lines[i].trim()) {
          html.push('<tr>' + cells(lines[i]).map((c) => `<td>${inline(c.trim())}</td>`).join('') + '</tr>');
          i++;
        }
        html.push('</tbody></table>');
        continue;
      }

      // List item
      const item = line.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
      if (item) {
        const depth = Math.floor(item[1].replace(/\t/g, '  ').length / 2) + 1;
        const kind = /^\d/.test(item[2]) ? 'ol' : 'ul';
        while (listStack.length > depth) closeLists(listStack.length - 1);
        while (listStack.length < depth) {
          // Honour the author's first number, so a list starting at 3 does.
          const from = kind === 'ol' ? parseInt(item[2], 10) : 1;
          html.push(kind === 'ul' ? '<ul>' : (from > 1 ? `<ol start="${from}">` : '<ol>'));
          listStack.push(kind);
        }
        html.push('<li>' + inline(item[3]) + '</li>');
        i++;
        continue;
      }

      // Blank line
      if (!line.trim()) {
        // A blank line between items makes one loose list, not several. Closing
        // unconditionally started a fresh <ol> per item, so every one of them
        // rendered as "1."
        const next = lines.slice(i + 1).find((l) => l.trim());
        const listContinues = listStack.length && next
          && /^(\s*)([-*+]|\d+[.)])\s+/.test(next);
        if (!listContinues) closeLists(0);
        i++;
        continue;
      }

      // Paragraph
      closeLists(0);
      const para = [];
      while (i < lines.length && lines[i].trim() &&
             !/^\s*(`{3,}|~{3,})/.test(lines[i]) &&
             !/^(#{1,6})\s/.test(lines[i]) &&
             !/^\s*([-*+]|\d+[.)])\s/.test(lines[i]) &&
             !/^\s*>/.test(lines[i])) {
        para.push(lines[i]);
        i++;
      }
      html.push('<p>' + inline(para.join('\n')).replace(/\n/g, '<br>') + '</p>');
    }

    closeLists(0);
    return html.join('\n');
  }

  global.md = { render, escapeHtml, highlight, highlightLines };
})(window);

function copyCode(button) {
  const block = button.closest('.code-block');
  const code = block ? block.dataset.code : '';
  navigator.clipboard.writeText(code).then(() => {
    button.classList.add('copied');
    showCopyToast();
    setTimeout(() => button.classList.remove('copied'), 1400);
  }).catch(() => {});
}
