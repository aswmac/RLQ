//
//  RLQApp.swift
//  RLQ
//
//  Created 2025.06.27.140857
//

import SwiftUI

@main
struct RLQApp: App {

	var body: some Scene {
		DocumentGroup(newDocument: RLQDocument()) { file in
			ContentView(document: file.$document)
		}
	}
}
