Visual design means an average reader gets the point from a glance, before reading a caption. It covers figures, diagrams, screenshots, tables, slides, posters, and the style layer of a page. It is not decoration, and it is not a brand system.

When two rules disagree, the rule that helps the reader understand wins. A test beats a preference. No single cue carries meaning alone.

These four phases set the order of an edit:
1. Condense. Cut the decoration and every mark that repeats another: the second gridline set, borders, shadows, 3D, and a legend that direct labels replace.
2. Expand. Put back what the cut removed wherever the reader needed it: units, a baseline, a source line, labels on the marks, and a caption that says what the reader is looking at.
3. Apply the rules below.
4. Verify. Run the tests in "Check your work" and fix what they find.

## Choose the form before you draw
- Use a picture only when it shows something words cannot: structure, comparison, sequence, or magnitude. A sentence or a table wins when it answers faster. (plain_rules, applied before the first mark)
- More than about twenty values form a shape worth a chart. Four values belong in a table, where the reader can read them exactly. (House rule, 19 November 2026)
- Write the claim the figure will make before you pick the chart. When you cannot write the claim, the figure has no job.

## Lead with the main point
- The title states the claim, not the topic: "Checkout fails twice as often on mobile Safari", not "Errors by browser". A topic title is allowed only when the figure has no single claim, such as a lookup chart or exploratory data. (Minto, "The Pyramid Principle"; USWDS data visualization guidance)
- One element wins the eye. When two compete, one of them is decoration. (Mayer, "Multimedia Learning", signaling principle)
- Give a dashboard a reading order. Each panel wins inside its frame, and the most important panel sits where the eye lands first.

## Group so the eye reads in order
- Group by proximity first, then enclosure. Put related marks closer than unrelated ones, because space costs nothing and says as much as a box.
- Nest one level. A box around every item stops meaning anything.

## Encode by position
- Readers judge position on a common scale most accurately, then length, then angle, then area, then color. Pick the encoding by the comparison the reader has to make. (Cleveland and McGill, "Graphical Perception", 1984)
- A length or an area starts at zero. A truncated axis makes a small change look large.
- One axis, one unit. Never two y-axes on one frame.
- Say what every mark is compared against. "Carries more" needs the thing it carries more than.

## Label directly
- Label the mark, not a legend, so the reader never looks a name up. A legend earns its place only when labels would collide. (NN/g, "Clutter-Free"; Mass.gov data visualization guidance)
- Put the unit next to the number, or on the axis.
- Use the reader's word in a label, not the field name or a code. "Active users last 7 days", not "usr_cnt_7d".
- Cut a label that repeats the title, and cut a legend that repeats direct labels.
- Put the source line under the figure it backs, not in an appendix.

## Use color as a cue, never as the only cue
- Contrast is not optional: 4.5:1 for text and 3:1 for marks and other non-text shapes. Contrast is the most common critical failure in published data visualizations, at about 88% of audits. (WCAG 1.4.3 and 1.4.11; Chartability)
- Cap the categories at about seven, and give each one a second cue, such as a direct label, a shape, or a position.
- Use a perceptually uniform ramp for continuous data. Never a rainbow map, which makes some changes look large and others small. (Crameri, Shephard, and Heron, "The misuse of colour in science communication", 2020; ColorBrewer)
- Change lightness, not only hue, when two colors must be told apart.

## Set the page and the type for reading
- This section applies to any page with style. Chat replies and plain markdown skip it.
- Keep the line length between 50 and 75 characters, ragged right, never justified.
- Set the leading two points larger than the type, as in 12 over 14.
- Use no more than three typefaces, and do not mix fonts in the body.
- Build hierarchy from size, weight, and position before color.
- Make headings left-aligned, in upper and lower case, and bold. Use the built-in list markup so a screen reader announces the list and its length.
- Use round or square bullets, at most two styles, and numbers only for a sequence.
- Add a picture only when it shows something the words cannot.

## Avoid the AI look
- This list serves taste rather than understanding, so every rule above outranks it. It is dated because the model settles on a new look, and this list was checked on 19 November 2026.
- Skip the defaults a model reaches for when it has made no decision: a purple-to-blue gradient, glassmorphism, three equal cards in a row, Inter on gray, gradient text, a sparkle icon, and a drop shadow on every card.
- Remove decoration that means nothing. This look costs the reader attention and makes every product look the same.

## Check your work
- Read the artifact at four scopes, widest first: whole surface, group, mark, label.
- Squint at it. Whatever still reads is what the eye finds first, and that must be the point.
- Desaturate it. If the meaning disappears, color was carrying it alone.
- Read only the titles. They should say the argument without the data.
- Shrink it to phone width, then to a thumbnail. If the point dies at either size, the figure is too busy.
- Follow the export. Check the PNG at the size the reader gets, the print in grayscale, and any copy pasted into a slide.
- Name the source of every number. A number with no source comes out of the figure.

## Take the fix from the table

| Instead of | Do this |
| --- | --- |
| A pie chart with eight slices | A sorted bar chart, or a table |
| A 3D bar, or an area chart for a linear value | A flat bar, or a line |
| A rainbow ramp | A perceptually uniform sequential ramp |
| A legend with six entries | Direct labels on the marks |
| An icon with no word | An icon and its label |
| A truncated axis on a bar chart | A zero baseline, or a line chart |
| A drop shadow, a gradient, a sparkle | Nothing |
