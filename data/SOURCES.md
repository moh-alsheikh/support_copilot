# Sample Knowledge Base Sources

Attribution and licensing for every file in `data/raw/`. This file lives outside `data/raw/` on
purpose so it is never ingested as a document. Downloaded 2026-09-30.

| File | Source | License | Changes |
|---|---|---|---|
| `raspberry-pi-5-product-brief.pdf` | [datasheets.raspberrypi.com](https://datasheets.raspberrypi.com/rpi5/raspberry-pi-5-product-brief.pdf) — © Raspberry Pi Ltd | CC BY-ND 4.0 | None |
| `27w-usb-c-power-supply-product-brief.pdf` | [datasheets.raspberrypi.com](https://datasheets.raspberrypi.com/power-supply/27w-usb-c-power-supply-product-brief.pdf) — © Raspberry Pi Ltd | CC BY-ND 4.0 | None |
| `raspberry-pi-active-cooler-product-brief.pdf` | [datasheets.raspberrypi.com](https://datasheets.raspberrypi.com/cooling/raspberry-pi-active-cooler-product-brief.pdf) — © Raspberry Pi Ltd | CC BY-ND 4.0 | None |
| `raspberry-pi-5-safety-and-user-guide.pdf` | [pip.raspberrypi.com](https://pip.raspberrypi.com/categories/1043-safety-info/documents/RP-006510-DS/Raspberry-Pi-5-e-label.pdf) (published as "Raspberry-Pi-5-e-label.pdf") — © Raspberry Pi Ltd | CC BY-ND 4.0 ¹ | Renamed only |
| `led-warning-flash-codes.md` | [raspberrypi/documentation](https://github.com/raspberrypi/documentation/blob/master/documentation/asciidoc/computers/configuration/led_blink_warnings.adoc) — © Raspberry Pi Ltd | CC BY-SA 4.0 | Saved as `.md` (content is unchanged AsciiDoc) |
| `power-supplies.md` | [raspberrypi/documentation](https://github.com/raspberrypi/documentation/blob/master/documentation/asciidoc/computers/raspberry-pi/power-supplies.adoc) — © Raspberry Pi Ltd | CC BY-SA 4.0 | Saved as `.md` (content is unchanged AsciiDoc) |
| `bootloader-diagnostics.png` | [raspberrypi/documentation](https://github.com/raspberrypi/documentation/blob/master/documentation/asciidoc/computers/raspberry-pi/images/bootloader-diagnostics.png) — © Raspberry Pi Ltd | CC BY-SA 4.0 | None |
| `gpio-pinout-diagram.png` | [raspberrypi/documentation](https://github.com/raspberrypi/documentation/blob/master/documentation/asciidoc/computers/raspberry-pi/images/GPIO-Pinout-Diagram-2.png) — © Raspberry Pi Ltd | CC BY-SA 4.0 | Renamed only |
| `cpsc-power-bank-recalls.json` | [US CPSC Recalls API](https://www.cpsc.gov/Recalls/CPSC-Recalls-Application-Program-Interface-API-Information), query `ProductName=power bank` | US federal government work, public domain ² | None |
| `cpsc-battery-charger-safety-meeting-1998-scan.pdf` | [CPSC meeting log, 2 April 1998](https://www.cpsc.gov/content/Safety-Issues-with-Batteries-Battery-Chargers-and-Charging-Methods) (published as `foia_battery1.pdf`), added 2026-10-01 | US federal government work, public domain ² | Renamed only |
| `support-tickets-sample.csv` | [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) (v11), © Bitext Innovations | CDLA-Sharing-1.0 | Subset: first 5 rows per intent (135 of 26,872); `flags` column dropped |

¹ Raspberry Pi's [licensing page](https://www.raspberrypi.com/licensing/) states most of its PDF
documentation is CC BY-ND 4.0; this file does not state its own license.

² The CPSC pages state no terms of use.

`cpsc-battery-charger-safety-meeting-1998-scan.pdf` is the knowledge base's only genuinely scanned
document: 3 image-only pages with no text layer (produced by "Acrobat 5.0 Image Conversion
Plug-in"), so every page goes through Poppler + Tesseract OCR. It is not Raspberry Pi content; it
sits with the power-bank recalls as general battery and charger safety material. Two quirks are
kept on purpose as realistic noise:

- Page 3 belongs to a **different** meeting log (fabric flammability testing). It was misfiled in
  the original file.
- A "received" stamp overlaps the meeting date on page 1, so OCR garbles the date.

`support-tickets-sample.csv` is synthetic, generic retail support data (orders, refunds,
accounts), not Raspberry Pi-specific. Under CDLA-Sharing-1.0, this modified subset must be shared
under the same license.
