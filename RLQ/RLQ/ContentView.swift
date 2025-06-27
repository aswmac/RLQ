//
//  ContentView.swift
//  RLQ
//
//  Created by Adam Mcgregor on 6/23/25.
//

import SwiftUI

struct ContentView: View {
    @Binding var document: RLQDocument

    var body: some View {
        MatrixView(matrix: document.mat)
    }
}

//#Preview {
//    ContentView(document: .constant(RLQDocument()))
//}
