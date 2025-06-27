//
//  RLQApp.swift
//  RLQ
//
//  Created by Adam Mcgregor on 6/23/25.
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
